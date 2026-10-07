"""The eval cycle's RECORD side — split out of ``eval_callback.py`` (2026-10-01).

Shared by BOTH eval callbacks so the bot-only and self-play paths surface identical keys: the
per-opponent TensorBoard/TUI recorder, the bot / externals metadata blocks, the anchored-BT ELO
(``record_elo``), the display-only external ELOs, and the RESUME side — re-publishing the last
recorded eval to the TUI and restoring the cadence anchor from metadata. ``eval_callback`` re-exports
every public name here; a test that STUBS ``send_metrics`` for the resume republish patches it HERE.
"""
import json
import math
import os

from agents.training.eval_schedule import RANDOM_OPPONENT_NAME
from agents.training.fixed_opponent_pool import is_external
from main.launcher.ipc import send_metrics


def read_latest_eval_block(path: str | None) -> dict | None:
    """Return the most recent eval block from a metadata.json, or None.

    `record_eval_results` stores it at the top level as `latest_eval`
    (build_bot_eval_block + step). Used to re-publish the last eval to the TUI
    after a restart. Falls back to the legacy per-checkpoint
    `snapshot_history[<ckpt>]["evals"]` layout for older metadata files.
    """
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return None
    latest = meta.get("latest_eval")
    if isinstance(latest, dict):
        return latest
    # Legacy fallback: evals nested under each checkpoint.
    best = None
    for entry in (meta.get("snapshot_history") or {}).values():
        ev = entry.get("evals") if isinstance(entry, dict) else None
        if isinstance(ev, dict) and (best is None or ev.get("step", -1) > best.get("step", -1)):
            best = ev
    return best


def replay_last_eval_to_tui(model_dir: str | None, resume_eval_metadata: str | None = None) -> None:
    """Re-publish the most recent persisted eval to the TUI on startup/resume.

    Reads ``latest_eval`` from this run's metadata.json (and the resumed checkpoint's, if
    given), and pushes the per-opponent + aggregate win-rate/reward/ep-len keys so the eval
    panel isn't blank until the next (possibly millions-of-steps-away) cycle. Shared by both
    eval callbacks.

    The self-play ``pool`` sub-block (aggregate + per-sentinel rows) is re-published too,
    so the Pool/sentinel rows survive a restart exactly like the bot rows. This is safe even
    though sentinels are positional and the pool slides: the pool only changes at an
    eval-collect (seed/promote), which is also when the block is persisted — so the saved
    rows match the pool that's reconstructed from ``snapshots/`` at restart. (Pre-seed evals
    persist an empty ``sentinels`` list; those aren't re-published — nothing to show yet.)
    """
    block = None
    for path in (
        os.path.join(model_dir, "metadata.json") if model_dir else None,
        resume_eval_metadata,
    ):
        block = read_latest_eval_block(path) or block
    if block is None:
        return
    opponents = block.get("opponents", {})
    if not opponents:
        return
    tui: dict[str, float] = {}
    for name, m in opponents.items():
        tui[f"eval/win_rate_vs_{name}"] = m.get("win_rate", 0.0)
        tui[f"eval/mean_ep_len_vs_{name}"] = m.get("mean_ep_len", 0.0)
        tui[f"eval/mean_reward_vs_{name}"] = m.get("mean_reward", 0.0)
    tui.update({
        "eval/win_rate_mean": block.get("win_rate_mean", 0.0),
        "eval/win_rate_vs_bots": block.get("win_rate_vs_bots", 0.0),
        "eval/mean_reward_mean": block.get("mean_reward_mean", block.get("mean_reward_vs_bots", 0.0)),
        "eval/mean_reward_vs_bots": block.get("mean_reward_vs_bots", 0.0),
        "eval/mean_ep_len_vs_bots": block.get("mean_ep_len_vs_bots", 0.0),
        "_step": block.get("step", 0),
    })
    # Self-play pool block: re-publish the aggregate + per-sentinel rows (newest→oldest,
    # positional sentinel_<i>) using the saved step tags, mirroring the live collect path.
    pool = block.get("pool")
    if isinstance(pool, dict) and pool.get("sentinels"):
        tui["eval/win_rate_vs_pool"] = pool.get("win_rate", 0.0)
        tui["eval/mean_reward_vs_pool"] = pool.get("mean_reward", 0.0)
        tui["eval/mean_ep_len_vs_pool"] = pool.get("mean_ep_len", 0.0)
        tui["eval/sentinel_monotonicity"] = pool.get("monotonicity", 1.0)
        if "snapshot_count" in pool:
            tui["eval/pool_snapshot_count"] = float(pool["snapshot_count"])
        for i, s in enumerate(pool["sentinels"]):
            tui[f"eval/win_rate_vs_sentinel_{i}"] = s.get("win_rate", 0.0)
            tui[f"eval/mean_reward_vs_sentinel_{i}"] = s.get("mean_reward", 0.0)
            tui[f"eval/mean_ep_len_vs_sentinel_{i}"] = s.get("mean_ep_len", 0.0)
            tui[f"eval/sentinel_step_{i}"] = float(s.get("step", 0))

    # Skill rating (ELO) — re-publish so the 🏅 badge + per-opponent ELO show immediately on
    # resume instead of blanking until the next (possibly millions-of-steps-away) eval cycle.
    # The saved HEADLINE elo is authoritative — set it FIRST so a fit failure below can never
    # drop it. Then fit the block (best-effort) to (a) compute the headline if the block predates
    # the `elo` field, and (b) recover each opponent's ELO for the panel.
    if "elo" in block:
        tui["eval/elo"] = block["elo"]
        tui["eval/elo_ci"] = block.get("elo_ci", 0.0)
    try:
        from agents.training import elo as elo_mod
        efit = elo_mod.fit_from_block(block)
        if efit is not None:
            if "elo" not in block:
                tr = efit.rating_for_step(int(block.get("step", 0)))
                if tr is not None:
                    tui["eval/elo"] = tr[0]
                    tui["eval/elo_ci"] = elo_mod.ci95(tr[1])
            sentinels = pool.get("sentinels", []) if isinstance(pool, dict) else []
            _record_opponent_elos(efit, opponents, sentinels, tui)
    except Exception as e:  # noqa: BLE001 — telemetry; never break resume
        print(f"⚠️ [ELO] resume-republish compute failed: {e}")
    send_metrics(tui)
    pool_note = (f", pool {pool['win_rate'] * 100:.1f}%"
                 if isinstance(pool, dict) and pool.get("sentinels") else "")
    print(f"[EVAL] resumed — re-published last eval (step {block.get('step', '?')}, "
          f"{block.get('win_rate_mean', 0.0) * 100:.1f}% mean{pool_note}) to the TUI")


def latest_recorded_eval_step(model_dir: str | None, resume_eval_metadata: str | None = None) -> int:
    """The most recent eval step recorded in metadata (this run's + the resumed checkpoint's),
    or 0 if none.

    Used to restore ``_last_eval_step`` on startup so a RESUMED run doesn't immediately re-eval
    the same checkpoint: ``_last_eval_step`` is in-memory and resets to 0 each process, and the
    resumed step is far past a cadence boundary, so a fresh 0 would fire an eval on the first
    step. Restoring the genuine last-eval step makes the cadence purely step-based — restarts
    neither duplicate the eval nor starve it (frequent restarts still eval once the step crosses
    the next boundary). The panel isn't left blank: ``replay_last_eval_to_tui`` re-publishes the
    numbers regardless.
    """
    step = 0
    for path in (
        os.path.join(model_dir, "metadata.json") if model_dir else None,
        resume_eval_metadata,
    ):
        blk = read_latest_eval_block(path)
        if blk:
            step = max(step, int(blk.get("step", 0)))
    return step


def bot_mean(d: dict[str, float]) -> float:
    """Average of values across the scripted BOTS only — excludes Random (the broken-model
    floor) AND any stable cross-run opponents (``ext_...``, a separate yardstick kept out of
    ``win_rate_vs_bots`` so it never moves the self-play curriculum or the bot aggregate)."""
    vals = [v for k, v in d.items() if k != RANDOM_OPPONENT_NAME and not is_external(k)]
    return sum(vals) / len(vals) if vals else 0.0


def record_per_opponent(logger, tui: dict, names, win_rates: dict,
                        reward_means: dict, ep_lens: dict) -> None:
    """Record ``eval/{win_rate,mean_reward,mean_ep_len}_vs_<name>`` (TB logger + the TUI dict) for
    each ``name`` present in ``win_rates``. The shared per-opponent recorder for BOTH eval callbacks
    (bots + stable opponents); pool sentinels are positional (``sentinel_<i>``) and recorded apart."""
    for name in names:
        if name not in win_rates:
            continue
        for metric, value in (("win_rate", win_rates[name]),
                              ("mean_reward", reward_means.get(name, 0.0)),
                              ("mean_ep_len", ep_lens.get(name, 0.0))):
            key = f"eval/{metric}_vs_{name}"
            logger.record(key, value)
            tui[key] = value


def external_aggregate(ext_wr: dict) -> "float | None":
    """Mean win rate over stable cross-run opponents — only meaningful (and only emitted) for a
    mini-league (2+); with a single one it would just duplicate that opponent's own row."""
    return sum(ext_wr.values()) / len(ext_wr) if len(ext_wr) > 1 else None


def external_elo(trainee_elo: float, win_rate: float) -> float:
    """A BALLPARK ELO for a stable cross-run opponent: invert the Bradley-Terry win probability from
    the trainee's (bot-anchored) rating and the trainee's win rate vs it —
    ``R_opp = R_trainee − (400/ln10)·logit(win_rate)``. A single-edge estimate (rough), but on the
    SAME bot-anchored scale as the rest of the eval ladder, so it's a meaningful ballpark. The stable
    opponent is deliberately NOT a player in the BT fit itself (no ladder distortion) — this is a
    display-only derivation. ``win_rate`` is clamped to keep the logit finite (a 100-game eval can't
    resolve a rate past ~±0.05 of the bounds anyway), capping the gap at ≈±676 ELO."""
    p = min(max(win_rate, 0.02), 0.98)
    return trainee_elo - (400.0 / math.log(10.0)) * math.log(p / (1.0 - p))


def record_external_elos(logger, tui: dict, trainee_elo: "float | None", ext_wr: dict,
                         source_elos: "dict | None" = None) -> None:
    """Record ``eval/elo_vs_<label>`` (display-only) for each stable opponent, so the eval table's
    elo column is populated for the ``ext_`` rows (the TUI reads ``eval/elo_vs_<opp>``).

    Prefers the opponent's **own recorded ELO** (``source_elos[label]`` — read from its run's
    ``metadata.json:latest_eval.elo``; a well-fit, bot-anchored rating). Falls back to a single-edge
    **ballpark** inverted from the live trainee rating + win rate (``external_elo``) only when the
    opponent carries no recorded ELO. ``trainee_elo`` may be ``None`` (no fit yet) — a carried ELO is
    still shown; a fallback-only opponent is skipped that cycle."""
    source_elos = source_elos or {}
    for lab, wr in ext_wr.items():
        carried = source_elos.get(lab)
        if carried is not None:
            e = round(carried)
        elif trainee_elo is not None:
            e = round(external_elo(trainee_elo, wr))
        else:
            continue  # no recorded ELO and no trainee rating yet → nothing to show this cycle
        logger.record(f"eval/elo_vs_{lab}", e)
        tui[f"eval/elo_vs_{lab}"] = e


def build_externals_block(ext_labels, win_rates: dict, reward_means: dict, ep_lens: dict) -> dict:
    """The ``metadata.json:latest_eval`` ``externals`` sub-block for stable opponents (display-only)."""
    return {
        lab: {"win_rate": win_rates[lab],
              "mean_reward": reward_means.get(lab, 0.0),
              "mean_ep_len": ep_lens.get(lab, 0.0)}
        for lab in ext_labels
    }


def build_bot_eval_block(
    win_rates: dict[str, float],
    reward_means: dict[str, float],
    ep_lens: dict[str, float],
    td_resid_tails: "dict[str, float] | None" = None,
) -> dict:
    """Build the standard bot-eval metrics dict for metadata.json (opponents last).

    ``td_resid_tails`` (#4, optional) maps opponent → TD-residual tail (CVaR); when present a
    ``td_resid_tail_mean`` headline (mean over the per-opponent tails) is added and each
    opponent's own tail is folded into its ``opponents[name]`` entry. Omitted entirely when no
    captured battles produced residuals, so the block is byte-identical to before when unused."""
    td_resid_tails = td_resid_tails or {}
    block = {
        "win_rate_mean": sum(win_rates.values()) / len(win_rates) if win_rates else 0.0,
        "mean_reward_mean": sum(reward_means.values()) / len(reward_means) if reward_means else 0.0,
        "win_rate_vs_bots": bot_mean(win_rates),
        "mean_reward_vs_bots": bot_mean(reward_means),
        "mean_ep_len_vs_bots": bot_mean(ep_lens),
        "opponents": {
            name: {
                "win_rate": win_rates[name],
                "mean_reward": reward_means[name],
                "mean_ep_len": ep_lens[name],
                **({"td_resid_tail": td_resid_tails[name]} if name in td_resid_tails else {}),
            }
            for name in win_rates
        },
    }
    if td_resid_tails:
        block["td_resid_tail_mean"] = sum(td_resid_tails.values()) / len(td_resid_tails)
    return block


def _record_opponent_elos(fit, bot_names, sentinels, tui):
    """Write per-opponent ELO into the TUI dict: each bot's anchored rating (``eval/elo_vs_<name>``)
    + each sentinel's rating positionally (``eval/elo_vs_sentinel_<i>``, matching the win-rate
    rows). Shared by the live ``record_elo`` and the resume republish so keys/format match. The
    single-cycle sentinel rating is anchored only via the (greedy) trainee, so it's a rough
    estimate — the offline `python -m main.elo` fit (full per-snapshot history) is canonical."""
    bot_r = fit.bot_ratings()
    for name in bot_names:
        br = bot_r.get(name)
        if br is not None:
            tui[f"eval/elo_vs_{name}"] = round(br[0])
    for i, s in enumerate(sentinels):
        step = s.get("step") if isinstance(s, dict) else None
        if step is None:
            continue
        sr = fit.rating_for_step(int(step))
        if sr is not None:
            tui[f"eval/elo_vs_sentinel_{i}"] = round(sr[0])


def record_elo(model_dir, step, bot_win_rates, sentinels, n_games, logger, tui,
               bot_td_tails=None, bot_counts=None, externals=None, sentinel_regime=None,
               mirrored_pairs=None):
    """Append this cycle's results to ``eval_results.jsonl``, refit anchored Bradley-Terry
    ELO, and record ``eval/elo`` + ``eval/elo_ci`` to the SB3 logger + the TUI dict.

    Shared by BOTH eval callbacks so the bot-only and self-play paths surface ELO
    identically. ``sentinels`` is ``[{"step", "win_rate", "counts"}, …]`` (``[]`` on the bot path;
    ``counts`` optional). ``sentinel_regime`` is the cycle's ``{"greedy", "symmetric_teams"}``
    opponent-regime stamp, passed straight through to the row — see ``append_eval_result_row``.
    ``mirrored_pairs`` (``None`` = an unmirrored cycle) is the cycle's per-opponent pentanomial block
    (:func:`mirrored_pairs_block`); it stamps the row's regime, and the live fit then reads ONLY rows
    of this cycle's regime (``elo.load_rows(mirrored=…)``), so a run that crossed the boundary never
    rates one era against the other's games. Returns ``(elo, ci_halfwidth)`` for the current snapshot, or ``None``. The live number
    is the best estimate from data SO FAR (batch-BT is global, so early points retro-adjust
    as more cycles land); ``python -m main.elo`` re-fits canonically offline. Best-effort —
    never raises into the eval path. The import is lazy to avoid any import cycle.

    Also records the two HODGE scalars beside ``eval/elo`` (``agents.training.hodge``) — the
    cyclic width of the trainee's matchup profile, measured over trainee×bot×bot triangles.
    Computed BEFORE the row is appended so the numbers ride in the row (offline replotting)
    and so an omitted read is counted there rather than lost."""
    if not model_dir:
        return None
    try:
        from agents.model.snapshot import append_eval_result_row
        from agents.training import elo as elo_mod
        from agents.training import hodge as hodge_mod

        hodge_block = hodge_mod.record_live_hodge(logger, step, bot_win_rates, sentinels,
                                                  n_games, bot_counts=bot_counts)
        append_eval_result_row(model_dir, step, n_games, bot_win_rates, sentinels,
                               bot_td_tails=bot_td_tails, bot_counts=bot_counts,
                               externals=externals, hodge=hodge_block,
                               sentinel_regime=sentinel_regime, mirrored_pairs=mirrored_pairs)
        # Refits the WHOLE accumulated ladder to read this snapshot's rating. Cheap at the
        # expected scale (tens of snapshots → ms); wrapped best-effort so it can never break eval.
        fit = elo_mod.fit_from_run(model_dir, source="log", mirrored=mirrored_pairs is not None)
        rating = fit.rating_for_step(step)
        if rating is None:
            return None
        elo_val, se = rating
        ci = elo_mod.ci95(se)
        logger.record("eval/elo", elo_val)
        logger.record("eval/elo_ci", ci)
        tui["eval/elo"] = elo_val
        tui["eval/elo_ci"] = ci
        # Per-opponent ELO for the eval panel: each bot's anchored rating + each sentinel's
        # rating (positional, matching the win-rate rows). TUI-only (no per-opponent TB clutter).
        _record_opponent_elos(fit, bot_win_rates, sentinels, tui)
        return elo_val, ci
    except Exception as e:  # noqa: BLE001 — ELO is telemetry; never break eval
        print(f"⚠️ [ELO] live rating failed at step {step}: {e}")
        return None


# ── MIRRORED TEAM PAIRS (`gen3_mirrored_pairs_v1`, `--eval-mirrored-pairs`) ─────────────────────────
def mirrored_pairs_block(merged: dict, *, bots, sentinels=(), externals=()) -> dict:
    """The row's ``mirrored_pairs`` block from a cycle's ``merged`` result: each opponent's pentanomial
    (``merged["pairs"]``), bots by name, sentinels by their TRAINING STEP (a sentinel label is positional
    and names a different snapshot every cycle), externals by label. ``sentinels`` is
    ``[(label, step), …]``."""
    from agents.training.mirrored_pairs import SCHEMA

    pairs = merged.get("pairs") or {}
    return {"schema": SCHEMA,
            "bots": {n: pairs[n] for n in bots if n in pairs},
            "sentinels": {str(st): pairs[lab] for lab, st in sentinels if lab in pairs},
            "externals": {lab: pairs[lab] for lab in externals if lab in pairs}}


def record_pair_scores(logger, tui: dict, merged: dict, keys, tag: str) -> "dict | None":
    """Record the PAIR-LEVEL score of the opponents ``keys`` pooled — ``eval/pair_score_vs_<tag>`` and its
    95% half-width ``eval/pair_score_ci_vs_<tag>`` (the pentanomial interval over PAIRS, never games), plus
    ``eval/pairs_vs_<tag>`` (the pair count). Returns the ``mirrored_pairs.summary`` block, or ``None`` when
    none of ``keys`` carried pairs (an unmirrored cycle records nothing)."""
    from agents.training import mirrored_pairs as MP

    pairs = merged.get("pairs") or {}
    counts = MP.pooled(pairs.get(k) for k in keys)
    if counts is None:
        return None
    block = MP.summary(counts)
    assert block is not None
    if block["score"] is not None:
        half = (block["score_ci95"][1] - block["score_ci95"][0]) / 2.0
        for key, val in ((f"eval/pair_score_vs_{tag}", block["score"]),
                         (f"eval/pair_score_ci_vs_{tag}", half),
                         (f"eval/pairs_vs_{tag}", float(block["n_pairs"]))):
            logger.record(key, val)
            tui[key] = val
    return block
