"""``main.ops.critic_read``'s REPORT — the markdown and the one-line ledger quote.

Split out for the same reason ``main.critic_gate_render`` is: the read is a pipeline of
subprocesses and bootstraps, and the thing that turns its numbers into sentences has no business
sharing a file with them. Everything here is pure — it takes the finished document and returns
text, reads nothing and writes nothing.

**The report's shape is load-bearing.** The headline deltas come FIRST, each with its interval
and its registered label; then the full tables under all three selection weightings with the
registered one starred; then the CONDITIONING section (added 2026-09-09 — the spread identity and
the two decodes-from-`V`); then each run on its own; then the ledger QUOTE; then every command and
every path. A reader who stops after the summary has the answer, and a reader who does not stop
can check it.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from main.ops import conditioning_meters as CM
from main.ops import critic_readouts as R
from main.ops import team_conditioning as TC
from main.ops import quota_match as QM

TOOL = "critic_read"
TOOL_VERSION = 4

#: the summary table's headline quantities. The first four are the 2026-09-08 registration's;
#: the last two are the CONDITIONING primaries added 2026-09-09, after three offline reads
#: established that the critic's defect is a conditioning failure in the win head — it emits one
#: near-marginal win probability regardless of opponent AND of its own team. An arm built against
#: that defect has to be read on the meters that measure it.
HEADLINES = ("gate.resolution.bot", "identity.bias.late (turn>=25)", "identity.turn_contrast",
             "gate.skill.bot", "cond.spread_ratio.t1_3", "cond.own_team_r2.t1")
#: identity strata, in report order. `ALL` first, then the turn buckets, then the opponent split.
IDENTITY_STRATA = ("ALL",) + R.TURN_BUCKETS + ("bot", "pool")
#: the three selection corrections every identity quantity is reported under.
WEIGHTINGS = ("raw", "pop", "ipw")
WEIGHTING_NOTE = {
    "raw": "unweighted over the cf_audit draw — the estimand arm A's committed +0.3089 was "
           "computed under, kept for comparability",
    "pop": "cf_audit's stratified draw recombined at the frame's own (decile, outcome) mass — "
           "corrects the SAMPLER against the trace tree",
    "ipw": "`pop` times 1/capture_rate(opponent, outcome) from the cycle's eval_manifest — "
           "RULE OF EVIDENCE 17, correcting the loss-enriched TREE against the eval population",
}


# --------------------------------------------------------------------------- rendering

def _f(x: Optional[float], nd: int = 4, sign: bool = True) -> str:
    if x is None:
        return "—"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if v != v:
        return "nan"
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def _ci(ci: Sequence[float], nd: int = 4) -> str:
    if ci is None or len(ci) != 2:
        return "—"
    return f"[{_f(ci[0], nd)}, {_f(ci[1], nd)}]"


def _label(row: Dict[str, Any]) -> str:
    q = row.get("qualifier")
    return f"**{row['label']}**" + (f" ({q})" if q else "")


def ledger_line(doc: Dict[str, Any]) -> str:
    """The one-line ledger QUOTE, in the registered form. Numbers only — no interpretation."""
    by = {r["key"]: r for r in doc["deltas"]}

    def part(key: str, name: str) -> str:
        r = by.get(key)
        if r is None:
            return f"{name} Δ — (not computed)"
        # A frame-sensitive row's quote says WHICH frame it was read on. A matched number and an
        # as-traced one are different measurements, and the 2026-09-09 retraction is exactly what
        # happens when a ledger line does not say which one it quotes.
        qm = r.get("quota_match") or {}
        tag = {"MATCHED": " [QUOTA-MATCHED]", "UNMATCHED": " [FRAMES UNMATCHED]"}.get(
            qm.get("status"), "")
        return f"{name} Δ {_f(r['delta'])} {_ci(r['ci'])} {r['label']}{tag}"
    a_step, c_step = doc["arm"]["step"], doc["control"]["step"]
    # A CROSS-STEP pair says so in the quote. The registered read is arm@10M vs control@10M; a
    # read against a different step is a different comparison and must never be quoted as if it
    # were the registered one.
    at = (f"{a_step / 1e6:.0f}M" if a_step == c_step
          else f"{a_step / 1e6:.0f}M vs {c_step / 1e6:.0f}M (CROSS-STEP)")
    # 🚨 An OFFLINE-GENERATED read carries its own marker for the same reason CROSS-STEP does: the
    # registered read is the LIVE 100-game cycle, and a quote of a 400-game replay pasted into the
    # ledger without this would be read as that. The games/sentinel spec rides along, because
    # "high power" is not a magnitude anyone can check later.
    gen = doc["arm"].get("generated_by") or {}
    if doc["arm"].get("generated"):
        at += (f" [OFFLINE-GENERATED: {gen.get('games_per_opponent', '?')} games x "
               f"{len(gen.get('bots') or [])}+{gen.get('sentinels_used', '?')} opponents, "
               f"{'full capture' if gen.get('capture') == 'ALL' else gen.get('capture')}]")
    return (f"{doc['arm']['run']} vs {doc['control']['run']} at {at}: " +
            " · ".join([part("gate.resolution.bot", "G1 bot"),
                        part("identity.bias.late (turn>=25)", "identity bias late"),
                        part("identity.turn_contrast", "turn-contrast"),
                        part("cond.spread_ratio.t1_3", "spread ratio t1-3"),
                        part("cond.own_team_r2.t1", "own-team R2 t1")]))


# --------------------------------------------------------------------------- the frame profiles

def _profiles_block(doc: Dict[str, Any]) -> str:
    """The two cycles' REALIZED per-opponent capture profiles, and what the read did about them.

    Printed in the HEADER, before a single number, because it decides whether the frame-sensitive
    conditioning rows below are a reading at all. A nominal quota is not a realized one: under
    battle-level work-stealing each shard unit carries ``max(1, ceil(quota / n_shards))``, so a
    nominal 5/10/5 lands on disk as 8 traced wins and up to 12 traced losses per opponent. These
    are the counts ON DISK.
    """
    qm = doc.get("quota_match")
    L: List[str] = []
    A = L.append
    A("### REALIZED capture profile — what is actually on disk, per opponent")
    A("")
    if not qm:
        A("> The conditioning block was not computed for both runs, so no frame profile is read "
          "and no row is quota-matched.")
        return "\n".join(L)
    A("| role | run | cap W/L/D per opponent | traced battles | W / L / D | opponents |")
    A("|---|---|---|---|---|---|")
    for role in ("arm", "control"):
        pr = (qm.get("profiles") or {}).get(role) or {}
        t = pr.get("totals") or {}
        A(f"| {role} | `{pr.get('run')}` | **{'/'.join(str(c) for c in pr.get('cap') or [])}** | "
          f"{pr.get('n_traced')} | {t.get('wins')} / {t.get('losses')} / {t.get('draws')} | "
          f"{len(pr.get('opponents') or {})} |")
    A("")
    plan = qm.get("plan") or {}
    A(f"**Frames:** {plan.get('why')}")
    A("")
    for note in plan.get("notes") or []:
        A(f"- {note}")
    if plan.get("notes"):
        A("")
    if not plan.get("needed"):
        A("> ✅ **SYMMETRIC.** Both sides carry the same realized cap, so every conditioning row "
          "is read AS TRACED and this report is what the tool printed before quota matching "
          "existed. Residual per-opponent differences below the cap are the two runs' own outcome "
          "mixes, not a selection asymmetry — matching them would shrink both frames for no "
          "power gain.")
    elif not qm.get("enabled"):
        A("> 🚨 **UNEQUAL FRAMES, AND `--no-quota-match` WAS PASSED.** Every FRAME-SENSITIVE "
          "conditioning row below is printed with a hard `UNMATCHED — not a reading` marker and "
          "**NO label**. A fitted decoder's out-of-fold score rises with the frame it was fit on, "
          "so on unequal frames neither DETECTED nor NOT DETECTED is a claim this report may "
          "make. Re-run without the flag.")
    else:
        caps = "/".join(str(c) for c in plan.get("caps") or [])
        A(f"> ⚖️ **UNEQUAL FRAMES — MATCHED.** The {' and '.join(plan.get('sides') or [])} side "
          f"is subsampled to caps **{caps}** per opponent over "
          f"{qm.get('seeds')} seeded draws, with the capture rates RECOMPUTED for each subsample "
          "so rule 17 still holds on the view actually read. The FRAME-SENSITIVE rows' labels are "
          "decided on the MATCHED delta; the as-traced value is printed beside them marked "
          "UNMATCHED and is never labelled. Nothing is copied or written — the subsamples are "
          "in-memory battle lists.")
    A("")
    return "\n".join(L)


def _frame_note(row: Dict[str, Any]) -> str:
    """The one-cell frame marker a conditioning row carries in every table it appears in."""
    qm = row.get("quota_match")
    if not qm:
        return "as traced"
    st = qm.get("status")
    if st == "MATCHED":
        v = (qm.get("variants") or {}).get("battle") or {}
        sides = (v.get("sides") or {})
        caps = next((s["caps"] for s in sides.values()), qm.get("caps")) or []
        return f"MATCHED {'/'.join(str(c) for c in caps)}"
    if st == "UNMATCHED":
        return "🚨 UNMATCHED"
    return "as traced (frames equal)"


#: the three rows the (A)/(B) separation is read from, in the order the sign table lists them.
AB_ROWS = ("cond.within_team_resolution.all", "cond.within_stratum_resolution.all",
           "cond.team_spread_ratio.t1_3")


def _cell_census_table(doc: Dict[str, Any]) -> List[str]:
    """Each side's OWN-TEAM CELL CENSUS — how many cells, and how much is inside them.

    🚨 Printed BESIDE the rows, not in an appendix, because a within-cell resolution cannot be
    read without it. The pool carries 719 teams, so a few-thousand-battle frame averages a handful
    of episodes per team; a binned resolution on cells that size is largely the binning's own
    (positively biased) noise. `n_cells`, the battles and states inside them and the MEDIAN
    per-cell N are what say whether the number is a measurement.
    """
    L: List[str] = ["**The OWN-TEAM CELL CENSUS** — the cells these rows are computed on "
                    f"(a team is a cell only with >= {CM.MIN_TEAM_BATTLES} battles; the strata "
                    f"are {TC.TEAM_STRATA} quantiles of the team's leave-one-battle-out win rate, "
                    "cut at equal battle mass):", ""]
    L.append("| role | cells | teams seen | battles in cells | states | median battles/cell | "
             "median states/cell | min–max battles/cell |")
    L.append("|---|---|---|---|---|---|---|---|")
    for role in ("arm", "control"):
        fr = ((doc[role].get("conditioning") or {}).get("frame") or {})
        cf = fr.get("cell_frames") or {}
        for key, name in (("cond.within_team_resolution.all", "team"),
                          ("cond.within_stratum_resolution.all", "stratum")):
            c = cf.get(key)
            if not c:
                L.append(f"| {role} · {name} | — | {fr.get('n_teams_seen', '—')} | — | — | — | "
                         "— | (row omitted) |")
                continue
            L.append(f"| {role} · {name} | {c.get('n_cells')} | {fr.get('n_teams_seen', '—')} | "
                     f"{c.get('n_battles')} | {c.get('n_states')} | "
                     f"{_f(c.get('median_battles_per_cell'), 1, sign=False)} | "
                     f"{_f(c.get('median_states_per_cell'), 1, sign=False)} | "
                     f"{_f(c.get('min_battles_per_cell'), 0, sign=False)}–"
                     f"{_f(c.get('max_battles_per_cell'), 0, sign=False)} |")
        ts = fr.get("team_spread_frame") or {}
        if ts:
            L.append(f"| {role} · between-team spread | {_f(ts.get('n_cells'), 0, sign=False)} | "
                     f"{fr.get('n_teams_seen', '—')} | "
                     f"{_f(ts.get('n_battles'), 0, sign=False)} | — | "
                     f"{_f(ts.get('median_battles_per_cell'), 1, sign=False)} | — | — |")
    L.append("")
    return L


def _ab_block(doc: Dict[str, Any]) -> str:
    """The (A) CONDITIONING vs (B) SUBSTITUTION separation — the whole reason these rows exist.

    Arm 8's ``V`` decodes its OWN TEAM at turn 1 where every control reads ~0. Two readings fit
    that: the critic CONDITIONS on its team (teams differ in strength; knowing which one you hold
    predicts better) or it SUBSTITUTES team identity for board state (right per team, blind inside
    one). They are told apart by decomposing the skill into a BETWEEN-team part and a WITHIN-team
    part — and by whether the team decode FALLS as the board fills in.
    """
    by = {r["key"]: r for r in doc["deltas"]}
    rows = [by[k] for k in AB_ROWS if k in by]
    diff = by.get(CM.OWN_TEAM_R2_DIFF)
    if not rows and diff is None:
        return ""
    L: List[str] = []
    A = L.append
    A("### (A) CONDITIONING or (B) SUBSTITUTION — the within/between decomposition of the "
      "own-team decode")
    A("")
    A("A critic whose `V` decodes its OWN TEAM can be doing either of two things, and the "
      "own-team R² row alone cannot tell them apart: **(A)** it CONDITIONS on its team — teams "
      "differ in strength, so knowing which one it holds predicts better, and inside a team it "
      "still discriminates by the board; or **(B)** it SUBSTITUTES team identity for board "
      "state — right on average per team, blind INSIDE one. The separation is the classic "
      "between/within decomposition of a forecast's skill, with the own team as the cell:")
    A("")
    A("| | within-team resolution | between-team spread | own-team R² t1 − late |")
    A("|---|---|---|---|")
    A("| **(A) conditioning** | not lower, ideally higher | UP | NEGATIVE — the board takes over "
      "as the game unfolds |")
    A("| **(B) substitution** | **DOWN** | UP | ~0 or POSITIVE — team identity still carries the "
      "forecast late |")
    A("| neither | flat | flat | flat |")
    A("")
    L.extend(_cell_census_table(doc))
    A("| row | frame | arm | control | **Δ** | 95% CI | verdict |")
    A("|---|---|---|---|---|---|---|")
    for r in rows + ([diff] if diff is not None else []):
        A(f"| {r['quantity']} · `{r['stratum']}` | {_frame_note(r)} | {_f(r['arm'])} | "
          f"{_f(r['control'])} | **{_f(r['delta'])}** | {_ci(r['ci'])} | {_label(r)} |")
    A("")
    d = {k: (by[k]["delta"] if k in by else None) for k in AB_ROWS}
    A(f"**Reading of the three signs:** {TC.reading_of(d[AB_ROWS[0]], d[AB_ROWS[1]], d[AB_ROWS[2]])}.")
    A("")
    A("> 🚨 **The between-team spread is an AMPLITUDE; the own-team R² is an ALIGNMENT.** The "
      "spread ratio compares `sd(mean V per team)` with `sd(that team's win rate)` — how far "
      "apart the head's per-team opinions are. The own-team R² is an out-of-fold MONOTONE decode, "
      "invariant to scale — whether those opinions are in the right ORDER. The two can move in "
      "opposite directions (a head that orders its teams correctly but under-disperses reads R² "
      "UP and spread DOWN), so the sign table above is read with both rows in hand and the R² "
      "row is never read alone.")
    A("")
    A("> 🚨 **The per-TEAM cells are small and the coarse row is the check on them.** With ~719 "
      "teams in the pool a few-thousand-battle frame leaves a handful of episodes per team, and a "
      "binned resolution inside a cell that size is largely the binning's own noise — which is "
      "*positively* biased, so a small per-team number is evidence of neither reading. The "
      "STRATUM row is the same estimator on cells hundreds of episodes deep. **Where the two "
      "disagree, believe the stratum row and say so.**")
    A("")
    if diff is not None:
        A(f"> 🚨 **`{CM.OWN_TEAM_R2_DIFF}` is PROVISIONAL and is never labelled DETECTED.** "
          f"{CM.METER_BY_KEY[CM.OWN_TEAM_R2_DIFF].provisional_why}.")
        A("")
    return "\n".join(L)


def _matched_detail(doc: Dict[str, Any]) -> str:
    """The FRAME-SENSITIVE rows in full: both matched rungs, and the as-traced value beside them.

    Three lines per row, and only the two MATCHED ones carry a label. The UNMATCHED line is the
    number the tool would have printed before 2026-09-09 and is kept visible precisely so the size
    of the artefact is on the page rather than in a ledger entry.
    """
    rows = [r for r in doc["deltas"]
            if r["family"] == "conditioning" and (r.get("quota_match") or {}).get("status")
            in ("MATCHED", "UNMATCHED")]
    if not rows:
        return ""
    L: List[str] = []
    A = L.append
    A("### The FRAME-SENSITIVE rows, on the equalised frames")
    A("")
    A("A row is matched because its ESTIMATOR moves with the size of the frame it is computed on, "
      "not because of anything in its name — the flag is declared on the meter "
      "(`main.ops.conditioning_meters.Meter.frame_sensitive`). Two mechanisms qualify: an "
      "out-of-fold score of a decoder **FIT** on the frame, and an **UNCORRECTED** second moment "
      "whose unsubtracted noise grows as the cells shrink. Everything else in the section above "
      "is a weighted mean, a difference of noise-corrected spreads, or an OLS over the pinned "
      "opponent roster — statistics whose expectation does not move with frame size given correct "
      "weights — and is read AS TRACED.")
    A("")
    A("| row | frame | arm | control | **Δ** | 95% CI | verdict |")
    A("|---|---|---|---|---|---|---|")
    for r in rows:
        qm = r["quota_match"]
        key = r["key"]
        if qm["status"] == "UNMATCHED":
            u = qm["unmatched"]
            A(f"| `{key}` | 🚨 UNMATCHED (as traced) | {_f(u['arm'])} | {_f(u['control'])} | "
              f"**{_f(u['delta'])}** | {_ci(u['ci'])} | **{QM.UNMATCHED_LABEL}** |")
            continue
        for rung, title in (("battle", "MATCHED · battle"), ("decoder", "MATCHED · decoder")):
            v = (qm.get("variants") or {}).get(rung)
            if v is None:
                continue
            sides = v.get("sides") or {}
            caps = next((s["caps"] for s in sides.values()), [])
            spread = next((s["spread"] for s in sides.values()), [float("nan")] * 2)
            nb = next((s["median_battles"] for s in sides.values()), float("nan"))
            db = next((s["median_decoder_battles"] for s in sides.values()), float("nan"))
            n_seeds = next((s["n_seeds"] for s in sides.values()), 0)
            A(f"| `{key}` | {title} {'/'.join(str(c) for c in caps)} "
              f"({nb:.0f} battles, {db:.0f} decoder battles, {n_seeds} seeds) | "
              f"{_f(v['arm'])} {_ci(spread)} | {_f(v['control'])} | **{_f(v['delta'])}** | "
              f"{_ci(v['ci'])} | {_label(v)} |")
        u = qm["unmatched"]
        A(f"| `{key}` | UNMATCHED (as traced) | {_f(u['arm'])} | {_f(u['control'])} | "
          f"**{_f(u['delta'])}** | {_ci(u['ci'])} | *no label — not a reading* |")
    A("")
    A("The subsampled side's value is the **across-seed median** and the bracket beside it is the "
      "**2.5/97.5 across-seed spread**, i.e. how much the answer depends on WHICH battles the cut "
      "kept. The Δ's interval is the **median seed's own battle-clustered bootstrap** differenced "
      "against the other side's — the seed count is odd, so the median is an exact order "
      "statistic and the point and the interval describe the same draw. The label is decided on "
      "the BATTLE-matched rung; the DECODER-matched rung is printed beside it because "
      f"`MIN_TEAM_BATTLES = {CM.MIN_TEAM_BATTLES}` makes the own-team decoder's frame a nonlinear "
      "function of team diversity, so equal battle counts can leave the richer side's decoder "
      "with FEWER battles than the poorer side's (62 against 104 in the 2026-09-09 read) — "
      "battle-matching is then unfair to it.")
    A("")
    return "\n".join(L)


def _population_block(doc: Dict[str, Any]) -> str:
    """What POPULATION each side's numbers describe, in words.

    🚨 Stated for a LIVE cycle too. "12 opponents x 100 games, traced under the outcome quota" is
    every bit as much a design choice as a generated cycle's, and a header that describes only the
    unusual side invites the reader to treat the other as the neutral default — which is how a
    frame difference gets read as a result.

    An OFFLINE-GENERATED side is called that in the header, and the block carries the warning that
    its rows do not belong in a table beside a 100-game read. `critic_read` REFUSES to form a delta
    across the boundary at all, so a report can only ever be all-live or all-offline; this says
    which, and says out loud that the two kinds of report are not rows of one table.
    """
    arm, ctl = doc["arm"], doc["control"]
    lines = ["**The POPULATION these numbers describe** — every conditioning and identity row "
             "is a statistic OF the traced frame, so the frame is part of the reading.", ""]
    lines.append("| role | population |")
    lines.append("|---|---|")
    for role, d in (("arm", arm), ("control", ctl)):
        lines.append(f"| {role} | {d.get('population', 'UNKNOWN')} |")
    if arm.get("generated") or ctl.get("generated"):
        gen = arm.get("generated_by") or ctl.get("generated_by") or {}
        lines.append("")
        lines.append(
            "> 🚨 **OFFLINE-GENERATED.** Both cycles were replayed from their saved checkpoints "
            f"by `main.ops.eval_trace_gen` — {gen.get('games_per_opponent', '?')} games per "
            f"opponent against {gen.get('sentinels_used', '?')} pool sentinels, "
            f"{'every battle traced' if gen.get('capture') == 'ALL' else gen.get('capture')}. "
            "**These rows are NOT comparable with a live 100-game read and must never sit in one "
            "table beside it**: they are a different number of games, possibly a different number "
            "of opponent cells, and a random sample rather than the live outcome quota's "
            "loss-enriched slice. Report them as their own table, against their own floor pair.")
        note = gen.get("reproducibility_note")
        if note:
            lines.append("")
            lines.append(f"> Reproducibility: {note}"
                         f" (seed {gen.get('seed')}, {gen.get('workers')} worker(s), "
                         f"concurrency {gen.get('concurrency')}).")
    return "\n".join(lines)


def render_md(doc: Dict[str, Any]) -> str:
    arm, ctl = doc["arm"], doc["control"]
    L: List[str] = []
    A = L.append
    A(f"# CRITIC READ — `{arm['run']}` vs `{ctl['run']}`")
    A("")
    A(f"The critic ladder's registered read (ledger 2026-09-08 *REGISTRATION · THE CRITIC "
      f"LADDER*), produced by `python -m main.ops.{TOOL}` v{TOOL_VERSION} at "
      f"{doc['generated_at']}. Every number is a DELTA, **arm − control**, with the difference "
      "of the two runs' independent battle-clustered bootstraps. Strength is NOT read.")
    A("")
    A("| role | run | cycle | battles | draw/timeout share | anchors | live |")
    A("|---|---|---|---|---|---|---|")
    for role, d in (("arm", arm), ("control", ctl)):
        an = d["identity"]["anchor"]
        share = ("NOT RECORDED" if d["draw_share"] is None
                 else f"{d['draw_share'] * 100:.1f}%")
        live = (f"yes pid {','.join(map(str, d['cycle']['live_pids']))}"
                if d["cycle"]["live_pids"] else "no")
        nb = d["gate"]["strata"].get("all", {}).get("n_battles", "—")
        A(f"| {role} | `{d['run']}` | `step_{d['step']}` | {nb} | {share} | "
          f"{an['reproduced']}/{an['issued']} ({an['rate'] * 100:.1f}%) | {live} |")
    A("")
    A(_population_block(doc))
    A("")
    A("**Which cycle, and WHY** — a read that silently took the previous cycle is a read of a "
      "different model than the caller believes (backlog 2026-09-09: an arm whose launcher "
      "process was still alive was read one cycle back, and nothing said so). `--step N` pins it.")
    A("")
    for role, d in (("arm", arm), ("control", ctl)):
        A(f"- **{role}** `{d['run']}` → `step_{d['step']}`: "
          f"{d['cycle'].get('why_read', d['cycle'].get('why', '—'))}")
    A("")
    A(_profiles_block(doc))
    if doc["floor"]["path"] is None:
        A(f"> 🚨 **{R.NO_FLOOR_NOTE}.** The ladder's replicate floor is the control-vs-control "
          "difference and does not exist until a second control replicate does. Every DETECTED "
          "below is a detection against ZERO and carries that qualifier.")
    else:
        A(f"> Floor: `{doc['floor']['path']}` — "
          f"{len(doc['floor']['floors'] or {})} keyed magnitudes.")
    A("")

    A("## SUMMARY — the headline deltas")
    A("")
    A("| quantity | frame | arm | control | **Δ (arm − control)** | 95% CI | verdict |")
    A("|---|---|---|---|---|---|---|")
    by = {r["key"]: r for r in doc["deltas"]}
    for key in HEADLINES:
        r = by.get(key)
        if r is None:
            A(f"| `{key}` | — | — | — | — | — | **NOT COMPUTED** |")
            continue
        A(f"| {r['quantity']} · `{r['stratum']}` | "
          f"{_frame_note(r) if r['family'] == 'conditioning' else '—'} | {_f(r['arm'])} | "
          f"{_f(r['control'])} | **{_f(r['delta'])}** | {_ci(r['ci'])} | {_label(r)} |")
    A("")
    A("Direction, stated so a sign cannot be misread: **resolution** and **skill** are HIGHER "
      "is better; **identity bias** is `V − p̂`, so POSITIVE means the head is OPTIMISTIC "
      "against its own Monte-Carlo continuation and a NEGATIVE delta is an improvement; the "
      "**turn-contrast** is `corr(turn,V) − corr(turn,MC)` and its target is ZERO, so a "
      "negative delta from a positive control moves toward the clock. The **spread ratio**'s "
      "target is ONE (for any calibrated critic the between-opponent spread of `V` equals that "
      "of the outcome), so a POSITIVE delta from a control below 1.0 is an improvement; the "
      "**own-team R²** and the **opponent-class AUC** are decodes FROM `V`, higher is better, "
      "with 0.0 and 0.5 the respective chance levels. The **within-team / within-stratum "
      "resolution** is DISCRIMINATION INSIDE a cell — higher is better, 0.0 is a forecast that "
      "says the same thing about every state of a team — and the **between-team spread ratio** "
      "is the own-team analogue of the opponent identity, target ONE. Their signs are read "
      "TOGETHER, in the (A)/(B) table of section 3.")
    A("")

    for family, title in (("gate", "## 1. RESOLUTION — the calibration gate's metrics, per stratum"),
                          ("identity", "## 2. IDENTITY — V against the Monte-Carlo continuation")):
        A(title)
        A("")
        A("| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |")
        A("|---|---|---|---|---|---|---|---|---|")
        for r in doc["deltas"]:
            if r["family"] != family:
                continue
            star = " ⭐" if r["registered"] else ""
            A(f"| {r['quantity']}{star} | `{r['stratum']}` | "
              f"`{r['weighting'] or 'capture-rate (gauge)'}` | {_f(r['arm'])} | "
              f"{_f(r['control'])} | **{_f(r['delta'])}** | {_ci(r['ci'])} | {r['n_draws']} | "
              f"{_label(r)} |")
        A("")
    A("⭐ = the REGISTERED estimand for that quantity. The others are the same statistic under a "
      "different selection correction, printed so the registered number is never the only one on "
      "the page.")
    A("")
    A("### the three weightings")
    A("")
    A("| name | what it corrects |")
    A("|---|---|")
    for k in WEIGHTINGS:
        A(f"| `{k}` | {WEIGHTING_NOTE[k]} |")
    A("")
    A(f"The gate rows carry no weighting column because the scaffolding gauge applies the "
      f"capture-rate correction itself, from the same `eval_manifest.json` rates — there is no "
      f"unweighted variant of a gate row. And {R.anchor_note()}")
    A("")

    A("## 3. CONDITIONING — does the head know WHO it is playing and WHOSE TEAM it holds?")
    A("")
    A("Promoted 2026-09-09 from two committed measurements — "
      "`measurements/winprob_mixture_diagnostic_2026-09-09/` (the between-opponent SPREAD "
      "IDENTITY and the bias-on-Elo slope) and `measurements/winprob_probe_read_2026-09-09/` "
      "(the own-team leave-one-battle-out win-rate target, battle-grouped folds, HT reweighting) "
      "— and computed here by `main.ops.conditioning_meters`, which both measurement directories "
      "can import.")
    A("")
    A("> 🚨 **The spread ratio's target is 1.0.** For ANY calibrated critic "
      "`E[V | opponent] = E[y | opponent]` exactly, so the between-opponent spread of `V` must "
      "EQUAL the between-opponent spread of the outcome. A head emitting one marginal win "
      "probability regardless of opponent reads near ZERO. The identity needs no strength axis, "
      "which is why it is the primary row and the Elo slope the secondary one.")
    A("")
    A("> 🚨 **A SPREAD RATIO CAN SIT BELOW ITS OWN INTERVAL, for TWO independent reasons, and "
      "neither is a defect in the bootstrap.** (1) The noise-corrected ratio is **CLAMPED** — "
      "each side's sampling variance is subtracted and a negative result floored at zero, a "
      "biased non-monotone operator — so a point estimate of 0.000 routinely carries a CI like "
      "[0.21, 0.53] (`winprob_head_refit_2026-09-09` §12 hazard 1). (2) A **resampled "
      "between-group variance is UPWARD BIASED**, so even the unclamped `ratio_raw`'s draws can "
      "centre above its point (the mixture diagnostic reports the same shape on its η² rows). "
      "**THE INTERVAL IS THE READ, and a 0.000 is not \"no spread\".** The unclamped, "
      "uncorrected `ratio_raw` is printed beside the clamped one as the monotone companion — "
      "never as a substitute. Both effects hit the arm and the control alike, so the DELTA is "
      "the quantity least disturbed by either; the per-run points below are the ones to read "
      "with the caveat in hand.")
    A("")
    for role, d in (("arm", arm), ("control", ctl)):
        c = d.get("conditioning") or {}
        fr = c.get("frame") or {}
        if not c:
            A(f"> **{role} — `{d['run']}`: the conditioning block was NOT computed.**")
            A("")
            continue
        A(f"**{role} — `{d['run']}` @ `step_{d['step']}`**: {fr.get('n_states', '—')} states / "
          f"{fr.get('n_battles', '—')} battles / {fr.get('n_opponents', '—')} opponents / "
          f"{fr.get('n_teams', '—')} trainee teams · manifest `selection_schema` "
          f"{fr.get('selection_schema')} · `max|values − win_probs|` = "
          f"{fr.get('max_abs_values_minus_winprobs')} · {fr.get('n_draw_battles_excluded', 0)} "
          f"draw/timeout battles excluded (no binary outcome) · strength axis: "
          f"{(c.get('strength') or {}).get('note', '—')}")
        A("")
    A("| quantity | stratum | frame | arm | control | **Δ** | 95% CI | n draws | verdict |")
    A("|---|---|---|---|---|---|---|---|---|")
    cond_rows = [r for r in doc["deltas"] if r["family"] == "conditioning"]
    if not cond_rows:
        A("| — | — | — | — | — | — | — | — | **NOT COMPUTED** |")
    for r in cond_rows:
        star = " ⭐" if r["registered"] else ""
        A(f"| {r['quantity']}{star} | `{r['stratum']}` | {_frame_note(r)} | {_f(r['arm'])} | "
          f"{_f(r['control'])} | **{_f(r['delta'])}** | {_ci(r['ci'])} | {r['n_draws']} | "
          f"{_label(r)} |")
    A("")
    ab = _ab_block(doc)
    if ab:
        A(ab)
    detail = _matched_detail(doc)
    if detail:
        A(detail)
    A("Each run's OWN point and interval, so a delta is never the only number on the page:")
    A("")
    A("| quantity | arm | 95% CI | control | 95% CI |")
    A("|---|---|---|---|---|")
    ac, cc = (arm.get("conditioning") or {}), (ctl.get("conditioning") or {})
    for key, quantity, stratum in CM.METERS:
        ap, cp = (ac.get("points") or {}), (cc.get("points") or {})
        aci, cci = (ac.get("ci") or {}), (cc.get("ci") or {})
        if key not in ap and key not in cp:
            continue
        A(f"| {quantity} · `{stratum}` | {_f(ap.get(key))} | {_ci(aci.get(key))} | "
          f"{_f(cp.get(key))} | {_ci(cci.get(key))} |")
    A("")
    om = {role: (d.get("conditioning") or {}).get("omitted") or {}
          for role, d in (("arm", arm), ("control", ctl))}
    if any(om.values()):
        A("**Rows OMITTED, with the reason** — an unsupported meter is never emitted as a NaN "
          "that reads like a measurement:")
        A("")
        for role, entries in om.items():
            for key, why in sorted(entries.items()):
                A(f"- `{key}` · {role}: {why}")
        A("")
    A(f"⚠️ **Recorded `V`, one cycle.** {CM.recorded_v_note()}")
    A("")
    A("⚠️ **The rows above that are NOT frame-sensitive are read AS TRACED, and that is not an "
      "oversight.** The noise-corrected spread ratio, the spread delta, the Elo slope, every gate "
      "row and every identity row are weighted MEANS, differences of noise-corrected spreads, or "
      "an OLS over the pinned opponent roster. A weighted mean's expectation does not depend on "
      "how many states entered it given correct weights — which rule 17's capture-rate IPW "
      "supplies — so frame SIZE moves their variance and not their expectation, and equalising "
      "the frames would cost power without removing a bias.")
    A("")
    A("⚠️ **No permutation null is run here.** The probe read's nulls on these very targets sit "
      "at R² ≈ 0.00 and AUC ≈ 0.50, so a near-zero own-team R² is a head that cannot be told "
      "from chance; but the DELTA is what this report licenses, and a per-run *detection* claim "
      "needs that measurement's own null, not this table.")
    A("")
    A("## 4. Each run on its own — the registered G1–G4 rows at the read cycle")
    A("")
    for role, d in (("arm", arm), ("control", ctl)):
        rows = d["gate"].get("critic_gate_rows_at_step")
        A(f"**{role} — `{d['run']}` @ step_{d['step']}** "
          f"(`main.critic_gate` verdict: `{json.dumps(d['gate'].get('critic_gate_verdict'))}`)")
        A("")
        if not rows:
            A("> 🚨 **`main.critic_gate` produced no calibration row at this step** — the "
              "registered G1-G4 verdicts are MISSING for this run, and the deltas below are the "
              "only gate evidence in this report.")
            if d["gate"].get("critic_gate_refusal"):
                A("")
                A("```")
                A(d["gate"]["critic_gate_refusal"])
                A("```")
            A("")
            continue
        A("| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | "
          "G1 | G2 | G3 | G4 |")
        A("|---|---|---|---|---|---|---|---|---|---|")
        for name, s in rows.items():
            A(f"| `{name}` | {'yes' if s.get('gated') else 'no'} | "
              f"{_f(s.get('resolution'), sign=False)} {_ci(s.get('resolution_ci'))} | "
              f"{_f(s.get('baseline_resolution'), sign=False)} | "
              f"{_f(s.get('delta_resolution'))} | "
              f"{_f(s.get('skill'))} {_ci(s.get('skill_ci'))} | "
              + " | ".join('✅' if s.get(g) else '❌'
                           for g in ("G1_resolution", "G2_reliability", "G3_ece", "G4_skill"))
              + " |")
        A("")
        m = d["identity"]["murphy"]["ipw"]
        A(f"Identity, capture-rate reweighted: {d['identity']['n_labels']} labels / "
          f"{d['identity']['n_battles']} battles / {d['identity']['n_rollouts']} rollouts · "
          f"Brier {m['brier']:.4f} = REL {m['reliability']:.4f} − RES {m['resolution']:.4f} + "
          f"UNC {m['uncertainty']:.4f} + WBV {m['within_bin_forecast_var']:.4f} "
          f"(resid {m['residual']:+.2e}) · base rate {m['base_rate']:.4f} · "
          f"**resolution is {m['resolution_cap_share'] * 100:.1f}% of the base-rate cap** · "
          f"corr(turn,V) {d['identity']['turn']['raw']['corr_turn_v']:+.4f} vs corr(turn,MC) "
          f"{d['identity']['turn']['raw']['corr_turn_mc']:+.4f}")
        A("")

    A("## 5. THE LEDGER LINE")
    A("")
    A("```")
    A(ledger_line(doc))
    A("```")
    A("")
    A("## 6. Provenance — every command and every path")
    A("")
    A("```bash")
    A("export PYTHONPATH=$PYTHONPATH:src")
    A(doc["invocation"])
    A("")
    for role, d in (("arm", arm), ("control", ctl)):
        A(f"# {role} — {d['run']}" + ("  [READOUT REUSED FROM CACHE]" if d["reused"] else ""))
        for c in d["commands"] or ["(no subprocess: the cached readout was reused)"]:
            A(f"  {c}")
    A("```")
    A("")
    A("| what | path |")
    A("|---|---|")
    for role, d in (("arm", arm), ("control", ctl)):
        A(f"| {role} run dir (READ-ONLY) | `{d['run_dir']}` |")
        A(f"| {role} trace dir | `{d['trace_dir']}` |")
        for k, v in d["paths"].items():
            A(f"| {role} {k} | `{v}` |")
    A(f"| this report | `{doc['out']}/critic_read.md` |")
    A(f"| machine-readable | `{doc['out']}/critic_read.json` |")
    A("")
    A(f"Bootstraps: identity {doc['params']['boot']} draws, gate {doc['params']['gate_boot']} "
      f"draws, seed {doc['params']['seed']}, unit = BATTLE, deltas = difference of INDEPENDENT "
      "bootstraps. Nothing was written under `models/`.")
    return "\n".join(L) + "\n"
