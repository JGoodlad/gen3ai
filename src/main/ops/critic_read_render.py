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

TOOL = "critic_read"
TOOL_VERSION = 2

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
        return f"{name} Δ {_f(r['delta'])} {_ci(r['ci'])} {r['label']}"
    a_step, c_step = doc["arm"]["step"], doc["control"]["step"]
    # A CROSS-STEP pair says so in the quote. The registered read is arm@10M vs control@10M; a
    # read against a different step is a different comparison and must never be quoted as if it
    # were the registered one.
    at = (f"{a_step / 1e6:.0f}M" if a_step == c_step
          else f"{a_step / 1e6:.0f}M vs {c_step / 1e6:.0f}M (CROSS-STEP)")
    return (f"{doc['arm']['run']} vs {doc['control']['run']} at {at}: " +
            " · ".join([part("gate.resolution.bot", "G1 bot"),
                        part("identity.bias.late (turn>=25)", "identity bias late"),
                        part("identity.turn_contrast", "turn-contrast"),
                        part("cond.spread_ratio.t1_3", "spread ratio t1-3"),
                        part("cond.own_team_r2.t1", "own-team R2 t1")]))


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
    A("**Which cycle, and WHY** — a read that silently took the previous cycle is a read of a "
      "different model than the caller believes (backlog 2026-09-09: an arm whose launcher "
      "process was still alive was read one cycle back, and nothing said so). `--step N` pins it.")
    A("")
    for role, d in (("arm", arm), ("control", ctl)):
        A(f"- **{role}** `{d['run']}` → `step_{d['step']}`: "
          f"{d['cycle'].get('why_read', d['cycle'].get('why', '—'))}")
    A("")
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
    A("| quantity | arm | control | **Δ (arm − control)** | 95% CI | verdict |")
    A("|---|---|---|---|---|---|")
    by = {r["key"]: r for r in doc["deltas"]}
    for key in HEADLINES:
        r = by.get(key)
        if r is None:
            A(f"| `{key}` | — | — | — | — | **NOT COMPUTED** |")
            continue
        A(f"| {r['quantity']} · `{r['stratum']}` | {_f(r['arm'])} | {_f(r['control'])} | "
          f"**{_f(r['delta'])}** | {_ci(r['ci'])} | {_label(r)} |")
    A("")
    A("Direction, stated so a sign cannot be misread: **resolution** and **skill** are HIGHER "
      "is better; **identity bias** is `V − p̂`, so POSITIVE means the head is OPTIMISTIC "
      "against its own Monte-Carlo continuation and a NEGATIVE delta is an improvement; the "
      "**turn-contrast** is `corr(turn,V) − corr(turn,MC)` and its target is ZERO, so a "
      "negative delta from a positive control moves toward the clock. The **spread ratio**'s "
      "target is ONE (for any calibrated critic the between-opponent spread of `V` equals that "
      "of the outcome), so a POSITIVE delta from a control below 1.0 is an improvement; the "
      "**own-team R²** and the **opponent-class AUC** are decodes FROM `V`, higher is better, "
      "with 0.0 and 0.5 the respective chance levels.")
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
    A("| quantity | stratum | arm | control | **Δ** | 95% CI | n draws | verdict |")
    A("|---|---|---|---|---|---|---|---|")
    cond_rows = [r for r in doc["deltas"] if r["family"] == "conditioning"]
    if not cond_rows:
        A("| — | — | — | — | — | — | — | **NOT COMPUTED** |")
    for r in cond_rows:
        star = " ⭐" if r["registered"] else ""
        A(f"| {r['quantity']}{star} | `{r['stratum']}` | {_f(r['arm'])} | {_f(r['control'])} | "
          f"**{_f(r['delta'])}** | {_ci(r['ci'])} | {r['n_draws']} | {_label(r)} |")
    A("")
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
