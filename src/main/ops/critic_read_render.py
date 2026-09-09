"""``main.ops.critic_read``'s REPORT — the markdown and the one-line ledger quote.

Split out for the same reason ``main.critic_gate_render`` is: the read is a pipeline of
subprocesses and bootstraps, and the thing that turns its numbers into sentences has no business
sharing a file with them. Everything here is pure — it takes the finished document and returns
text, reads nothing and writes nothing.

**The report's shape is load-bearing.** The four headline deltas come FIRST, each with its
interval and its registered label; then the full tables under all three selection weightings with
the registered one starred; then each run on its own; then the ledger QUOTE; then every command
and every path. A reader who stops after the summary has the answer, and a reader who does not
stop can check it.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from main.ops import critic_readouts as R

TOOL = "critic_read"
TOOL_VERSION = 1

#: the four the summary table leads with — the registration's headline quantities.
HEADLINES = ("gate.resolution.bot", "identity.bias.late (turn>=25)", "identity.turn_contrast",
             "gate.skill.bot")
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
                        part("identity.turn_contrast", "turn-contrast")]))


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
    if doc["floor"]["path"] is None:
        A(f"> 🚨 **{R.NO_FLOOR_NOTE}.** The ladder's replicate floor is the control-vs-control "
          "difference and does not exist until a second control replicate does. Every DETECTED "
          "below is a detection against ZERO and carries that qualifier.")
    else:
        A(f"> Floor: `{doc['floor']['path']}` — "
          f"{len(doc['floor']['floors'] or {})} keyed magnitudes.")
    A("")

    A("## SUMMARY — the four headline deltas")
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
      "negative delta from a positive control moves toward the clock.")
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

    A("## 3. Each run on its own — the registered G1–G4 rows at the read cycle")
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

    A("## 4. THE LEDGER LINE")
    A("")
    A("```")
    A(ledger_line(doc))
    A("```")
    A("")
    A("## 5. Provenance — every command and every path")
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
