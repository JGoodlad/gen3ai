"""cf_audit_render — the bias map's markdown presentation, and nothing else.

Pure presentation: a bias-map dict in, a markdown string out. It computes no statistic and reads
no file — every number it prints was decided by ``cf_audit``'s readouts — so the two rules it does
carry are formatting ones, and both are load-bearing rather than cosmetic. An ABSENT evidential
block renders a one-line note instead of a row of zeros ("this checkpoint has no head" and "this
head claims no uncertainty" are opposite findings), and a flat width renders ``n/a`` rather than 0
for the same reason.

Extracted verbatim from ``cf_audit.py`` (2026-09-06, the file-size ratchet's second cut of the
1,000-2,000 band). ``cf_audit`` re-imports :func:`render_markdown`, so
``from agents.training.cf_audit import render_markdown`` still resolves, and the output is
unchanged: ``cf_audit_test.py``'s extraction-parity golden pins the rendered markdown itself
(captured BEFORE the move, reproduced byte-for-byte after).
"""

from __future__ import annotations

import json
import math
from typing import Optional


def render_markdown(bm: dict, *, run_dir: str, step: Optional[int], ckpt: Optional[str]) -> str:
    L = []
    ap = L.append
    h = bm.get("headline") or {}
    ap("# cf_audit — the counterfactual bias map\n")
    ap(f"**Run:** `{run_dir}`  ·  **step:** {step}  ·  **checkpoint:** `{ckpt}`  ·  "
       f"**R:** {bm['n_rollouts']}  ·  **sampler:** `{bm['sampler']['sampler_version']}` "
       f"(seed {bm['sampler']['seed']})\n")
    ap("## Headline\n")
    ap("```")
    ap(f"labels                          {h.get('n_labels')}   over {h.get('n_battles')} battles")
    ap(f"population-weighted gap         {h.get('population_weighted_gap'):+.4f}"
       if h.get("population_weighted_gap") is not None else "population-weighted gap  n/a")
    ap(f"population-weighted sd_true_excess  {h.get('population_weighted_sd_true_excess'):.4f}"
       if h.get("population_weighted_sd_true_excess") is not None else "sd_true_excess  n/a")
    ap("```")
    ap("\nThe **gap** is the offset a re-centring would fix; the **sd_true_excess** is the "
       "per-state spread the head does not resolve, and it is the primary meter. A lever that "
       "moves the first and not the second has not done the thing this program is for.\n")

    acc = bm.get("accounting") or {}
    ap("## Accounting\n")
    ap("| | |\n|---|---|")
    for k in ("frame_decisions", "frame_battles", "tasks_issued", "labelled", "errors",
              "anchors_issued", "anchors_reproduced", "rollouts"):
        if k in acc:
            ap(f"| {k.replace('_', ' ')} | {acc[k]} |")
    if acc.get("skipped"):
        ap(f"| skipped (frame) | {json.dumps(acc['skipped'])} |")
    ap("")

    def _tab(title, rows, cols=("stratum", "n", "n_battles", "mean_predicted", "mean_mc", "mean_gap")):
        if not rows:
            return
        ap(f"## {title}\n")
        ap("| " + " | ".join(cols) + " | 95% CI (battle-clustered) |")
        ap("|" + "---|" * (len(cols) + 1))
        for r in rows:
            ci = r.get("gap_ci") or [None, None]
            cells = []
            for c in cols:
                v = r.get(c)
                cells.append(f"{v:+.4f}" if isinstance(v, float) else str(v))
            ci_s = (f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
                    if ci[0] is not None else "—")
            ap("| " + " | ".join(cells) + f" | {ci_s} |")
        ap("")

    _tab("By battle outcome (a description of two state POPULATIONS, not a calibration verdict)",
         bm.get("by_outcome"))
    _tab("By predicted decile × outcome", bm.get("by_decile_outcome"))
    _tab("By turn tercile", bm.get("by_turn_tercile"))
    _tab("By opponent", bm.get("by_opponent"))

    cc = bm.get("conviction_class")
    if cc:
        ap("## The conviction class — high confidence, lost battle\n")
        ap("```")
        ap(f"n={cc['n']} over {cc['n_battles']} battles")
        ap(f"predicted {cc['mean_predicted']:.3f} (median {cc['median_predicted']:.3f})  "
           f"vs tight-MC {cc['mean_mc']:.3f} (median {cc['median_mc']:.3f})")
        def _ci(v):
            return ("—" if not v or v[0] is None else f"[{v[0]:+.4f}, {v[1]:+.4f}]")
        ap(f"gap {cc['mean_gap']:+.4f}  CI {_ci(cc.get('gap_ci'))}")
        if cc.get("loss_minus_win_ci"):
            ap(f"LOSS - WIN difference {cc['loss_minus_win_gap']:+.4f}  "
               f"CI {_ci(cc['loss_minus_win_ci'])}")
        ap(f"MC >= 0.75 (the critic was RIGHT; the dice lost it)  {cc['share_mc_ge_0.75'] * 100:.1f}%")
        ap(f"MC <  0.50 (the critic was genuinely wrong)          {cc['share_mc_lt_0.50'] * 100:.1f}%")
        ap(f"MC <  0.25 (badly wrong)                             {cc['share_mc_lt_0.25'] * 100:.1f}%")
        ap("```")
        ap("\nA single realized outcome cannot separate those two populations. That separation "
           "is the whole case for a tight-MC label as an instrument.\n")

    res = bm.get("resolution")
    has_evid = bool(res) and any(r.get("evid_width_mean") is not None for r in res)
    if res:
        ap("## RESOLUTION — within-decile true spread vs the binomial floor\n")
        cols = ("| decile | n | predicted | MC | sd(MC) | binomial floor | "
                "**sd_true_excess** | % variance real |")
        rule = "|---|---|---|---|---|---|---|---|"
        if has_evid:
            cols += " Beta width | Beta precision |"
            rule += "---|---|"
        ap(cols)
        ap(rule)
        for r in res:
            fv = f"{r['frac_variance_real'] * 100:.1f}%" if r.get("frac_variance_real") else "—"
            line = (f"| {r['decile']} | {r['n']} | {r['mean_predicted']:.3f} | {r['mean']:.3f} | "
                    f"{r['sd_observed']:.3f} | {r['sd_binomial_floor']:.3f} | "
                    f"**{r['sd_true_excess']:.3f}** | {fv} |")
            if has_evid:
                w, p = r.get("evid_width_mean"), r.get("evid_precision_mean")
                line += (f" {w:.3f} |" if w is not None else " — |")
                line += (f" {p:.2f} |" if p is not None else " — |")
            ap(line)
        ap("")

    ev = bm.get("evidential")
    ap("## EVIDENTIAL — does the confessed width track the blur?\n")
    if not ev:
        # A one-line NOTE, never a row of zeros: "this checkpoint has no head" and "this head
        # claims no uncertainty" are opposite findings and must not render the same.
        ap("_The audited checkpoint carries no `cf_evid_head` (`--cf-evidential` off, or pre-v98) —"
           " the evidential columns are ABSENT, not zero._\n")
    else:
        rho, ci = ev.get("width_vs_blur_spearman"), ev.get("width_vs_blur_ci") or [None, None]
        ap("```")
        ap(f"labels scored                   {ev.get('n_labels_scored')}   over "
           f"{ev.get('n_strata')} strata")
        ap(f"mean Beta width (epistemic sd)  "
           f"{ev['evid_width_mean']:.4f}" if ev.get("evid_width_mean") is not None else
           "mean Beta width  n/a")
        ap(f"mean Beta precision (alpha+beta)    "
           f"{ev['evid_precision_mean']:.3f}" if ev.get("evid_precision_mean") is not None else
           "mean Beta precision  n/a")
        ap(f"width_vs_blur_spearman          "
           f"{rho:+.3f}" if rho is not None else
           "width_vs_blur_spearman          n/a (flat width, or <3 strata)")
        if ci[0] is not None:
            ap(f"  95% CI (battle-clustered)     [{ci[0]:+.3f}, {ci[1]:+.3f}]  "
               f"over {ev.get('draws_usable')} usable draws")
        ap("```")
        ap("\nThe head reads the same `value_pooled` as the scalar one, so it cannot REMOVE the "
           "blur — only confess it. Success is this correlation, not a falling `nll`: **wide "
           "everywhere and wide nowhere are the same null**, and a flat width reports `n/a` rather "
           "than 0 so the two cannot be confused.\n")
    ap("## Caveats\n")
    ap("- Turn-1 decisions are excluded by construction (the offline replay driver cannot open "
       "them) and forced-switch rounds are structurally uncovered by the re-roll anchor.")
    ap("- The MC label is measured on the EVAL distribution, played greedy; the head was "
       "trained on a mostly-self-play mixture with a stochastic actor. Never quote a gap "
       "without naming the population — its SIGN depends on the weighting.")
    ap(f"- R = {bm['n_rollouts']}: a single label's own sd is at most "
       f"{0.5 / math.sqrt(bm['n_rollouts']):.3f} (95% half-width "
       f"±{1.96 * 0.5 / math.sqrt(bm['n_rollouts']):.2f}). Cell aggregates are honest; a single "
       "state's label is not a point value.")
    return "\n".join(L) + "\n"
