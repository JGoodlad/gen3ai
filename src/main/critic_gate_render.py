"""main.critic_gate_render — the two renderings of a `main.critic_gate` report.

The markdown one is what a ledger entry quotes; the text one is what a terminal shows. They carry
the SAME content deliberately — a reader who saw the console and a reader who opened the file must
not come away with different caveats, and the caveats are the point of this report.

Split out of ``critic_gate.py`` so the measurement half stays under the 1,000-line target; the
sibling convention is ``main/launcher/format.py``'s. Every number here is formatted, never
computed: the meter's own block is rendered by the METER's own renderer for the same reason.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List

from main.critic_gate_design import DESIGN_DOC, FALSIFICATION_CLAUSE, NOT_RUNNABLE


def _f(v: Any, spec: str = ".4f", dash: str = "—") -> str:
    if v is None:
        return dash
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    return dash if not math.isfinite(x) else format(x, spec)


def _tick(ok: Any) -> str:
    return "✅" if ok else "❌"


def _meter_digest(result: Dict[str, Any]) -> str:
    """The meter's OWN rendering, so its numbers are never re-formatted by a second opinion."""
    try:
        from agents.training import untaught_meter as engine
        return engine.render_text(result)
    except Exception:  # noqa: BLE001 — a digest is never worth failing the report for
        return json.dumps(result.get("result", result), indent=1)[:4000]


def render_markdown(doc: Dict[str, Any]) -> str:
    out: List[str] = []
    m = doc["_meta"]
    out.append(f"# CRITIC GATE — `{m['run']['run_base']}` vs `{m['parent']['run_base']}`")
    out.append("")
    out.append(f"The pre-registered read of `{DESIGN_DOC}` §5.5 (endpoints) / §4.3 (bars). "
               f"Generated {m['volatile']['generated_at']}.")
    out.append("")
    out.append("| input | spec | resolved file | rung |")
    out.append("|---|---|---|---|")
    for label, d in ([("run", m["run"]), ("parent", m["parent"])]
                     + [("control", c) for c in m["controls"]]):
        out.append(f"| {label} | `{d['spec']}` | `{d['resolved_file']}` | "
                   f"`{d['resolution_rung']}` |")
    out.append("")
    v = doc["verdict"]
    out.append(f"## VERDICT — **{v['verdict']}**")
    out.append("")
    out.append(f"> {v['why']}")
    out.append("")

    lad = doc.get("ladder")
    out.append("## 1. Anchored ladder, at matched SNAPSHOT COUNT")
    out.append("")
    if lad:
        out.append(f"**{lad['rating_note']}**")
        out.append("")
        out.append(f"| # | {m['run']['run_base']} step | elo ±95% | "
                   f"{m['parent']['run_base']} step | elo ±95% |")
        out.append("|---|---|---|---|---|")
        for a, b in zip(lad["run"]["nodes"], lad["parent"]["nodes"]):
            out.append(f"| {a['i']} | {a['step']:,} | {a['elo']:.0f} ± {a['ci95']:.0f} | "
                       f"{b['step']:,} | {b['elo']:.0f} ± {b['ci95']:.0f} |")
        out.append("")
        out.append(f"**Δ at {lad['at_snapshots']} snapshots: {lad['delta_elo']:+.0f} ELO "
                   f"[{lad['delta_ci95'][0]:+.0f}, {lad['delta_ci95'][1]:+.0f}]** — "
                   f"{lad['comparability']}")
    else:
        out.append("_not read_")
    out.append("")

    cal = doc.get("calibration")
    out.append("## 2. Calibration gate (G1–G4) — RESOLUTION is primary")
    out.append("")
    if cal:
        out.append(f"Bars read from `{cal['artifact']}` (`{cal['baseline_run']}`, steps "
                   f"{cal['baseline_steps']}, reduce=`{cal['baseline_reduce']}`), "
                   f"selection-reweighted. {cal['asymmetry_note']}")
        out.append("")
        out.append("| step | stratum | gated | **resolution** [95% CI] | baseline | Δ | "
                   "reliability | ECE | skill [95% CI] | G1 | G2 | G3 | G4 |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for c in cal["checkpoints"]:
            for s in c["strata"]:
                out.append(
                    f"| {c['step']:,} | `{s['stratum']}` | {'yes' if s['gated'] else 'no'} | "
                    f"**{_f(s['resolution'])}** [{_f(s['resolution_ci'][0])}, "
                    f"{_f(s['resolution_ci'][1])}] | {_f(s['baseline_resolution'])} | "
                    f"{_f(s['delta_resolution'], '+.4f')} | {_f(s['reliability'])} | "
                    f"{_f(s['ece'])} | {_f(s['skill'], '+.3f')} [{_f(s['skill_ci'][0], '+.3f')}, "
                    f"{_f(s['skill_ci'][1], '+.3f')}] | {_tick(s['G1_resolution'])} | "
                    f"{_tick(s['G2_reliability'])} | {_tick(s['G3_ece'])} | "
                    f"{_tick(s['G4_skill'])} |")
        out.append("")
        out.append(f"_{cal['not_gated_note']}_")
    else:
        out.append("_not read_")
    out.append("")

    k = doc.get("kill")
    out.append("## 3. G7 — stall rate + episode length (KILL condition)")
    out.append("")
    if k:
        out.append(f"Thresholds: stall rate ≤ {k['thresholds']['max_stall_rate']} "
                   f"(a battle at ≥ {k['thresholds']['stall_turns']} turns is a stall), episode "
                   f"length ≤ {k['thresholds']['max_ep_len_ratio']}× the era's. "
                   f"({k['thresholds']['threshold_provenance']}.)")
        out.append("")
        out.append("| step | stall rate (captured) | mean turns | ep_len bots | ep_len pool | "
                   "verdict |")
        out.append("|---|---|---|---|---|---|")
        for c in k["cycles"]:
            out.append(f"| {c['step']:,} | {_f(c['stall_rate_captured'])} | "
                       f"{_f(c['mean_turns_captured'], '.1f')} | {_f(c['ep_len_bots'], '.2f')} | "
                       f"{_f(c['ep_len_pool'], '.2f')} | "
                       f"{'**KILL** — ' + '; '.join(c['breaches']) if c['kill'] else 'OK'} |")
        out.append("")
        out.append(f"_{k['sources']['stall_rate']}_")
    else:
        out.append("_not read_")
    out.append("")

    mt = doc.get("untaught_meter")
    out.append("## 4. Untaught meter — with a CONTINUATION control")
    out.append("")
    if mt:
        out.append(f"`{' '.join(mt['argv'][1:])}`")
        out.append("")
        out.append(f"exit {mt['returncode']} · {mt['seconds']}s · {mt['control_note']}")
        if mt.get("result"):
            out.append("")
            out.append("```")
            out.append(_meter_digest(mt["result"]))
            out.append("```")
    else:
        out.append("_skipped (`--skip-meter`)_")
    out.append("")

    out.append("## Not runnable here")
    out.append("")
    out.append("| # | criterion | why |")
    out.append("|---|---|---|")
    for gid, what, why in NOT_RUNNABLE:
        out.append(f"| {gid} | {what} | {why} |")
    out.append("")
    out.append(f"*Falsification clause (design §5.5, verbatim): {FALSIFICATION_CLAUSE}*")
    return "\n".join(out)


def render_text(doc: Dict[str, Any]) -> str:
    """The console read. Same content as the markdown, shaped for a terminal."""
    w = 100
    m = doc["_meta"]
    out = ["=" * w,
           f"CRITIC GATE  —  {m['run']['run_base']}  vs  {m['parent']['run_base']}",
           f"  design {DESIGN_DOC} §5.5 (endpoints) / §4.3 (bars)", "=" * w, ""]
    for label, d in ([("run", m["run"]), ("parent", m["parent"])]
                     + [("control", c) for c in m["controls"]]):
        out.append(f"  {label:<8} {d['resolved_file']}  [rung={d['resolution_rung']} "
                   f"rule={d['resolution_rule']}]")
    v = doc["verdict"]
    out += ["", f"VERDICT: {v['verdict']}", f"  {v['why']}", ""]

    lad = doc.get("ladder")
    out.append("(1) ANCHORED LADDER — matched SNAPSHOT COUNT, never matched step")
    if lad:
        out.append(f"    {lad['rating_note']}")
        out.append(f"    {'#':>3}  {'run step':>14}{'elo':>8}{'±95%':>7}   "
                   f"{'parent step':>14}{'elo':>8}{'±95%':>7}")
        for a, b in zip(lad["run"]["nodes"], lad["parent"]["nodes"]):
            out.append(f"    {a['i']:>3}  {a['step']:>14,}{a['elo']:>8.0f}{a['ci95']:>7.0f}   "
                       f"{b['step']:>14,}{b['elo']:>8.0f}{b['ci95']:>7.0f}")
        out.append(f"    Δ at {lad['at_snapshots']} snapshots: {lad['delta_elo']:+.0f} ELO "
                   f"[{lad['delta_ci95'][0]:+.0f}, {lad['delta_ci95'][1]:+.0f}]")
        out.append(f"    {lad['comparability']}")
    else:
        out.append("    not read")
    out.append("")

    cal = doc.get("calibration")
    out.append("(2) CALIBRATION GATE — RESOLUTION primary; bot and pool NEVER pooled")
    if cal:
        out.append(f"    bars: {os.path.basename(cal['artifact'])} ({cal['baseline_run']}, "
                   f"reduce={cal['baseline_reduce']})")
        out.append(f"    {'step':>12}{'stratum':>9}{'btl':>5}{'resolution':>11}"
                   f"{'[ 95% CI ]':>20}{'base':>8}{'rel':>8}{'ece':>7}{'skill':>8}"
                   f"   G1 G2 G3 G4")
        for c in cal["checkpoints"]:
            for s in c["strata"]:
                flags = " ".join((" Y" if s[k] else " n") for k in
                                 ("G1_resolution", "G2_reliability", "G3_ece", "G4_skill"))
                ci = f"[{_f(s['resolution_ci'][0])},{_f(s['resolution_ci'][1])}]"
                out.append(f"    {c['step']:>12,}{s['stratum']:>9}{s['n_battles']:>5}"
                           f"{_f(s['resolution']):>11}{ci:>20}"
                           f"{_f(s['baseline_resolution']):>8}{_f(s['reliability']):>8}"
                           f"{_f(s['ece'], '.3f'):>7}{_f(s['skill'], '+.3f'):>8}  {flags}"
                           + ("" if s["gated"] else "   (not gated)"))
        out.append(f"    {cal['not_gated_note']}")
        out.append(f"    {cal['asymmetry_note']}")
    else:
        out.append("    not read")
    out.append("")

    k = doc.get("kill")
    out.append("(3) G7 KILL CONDITION — stall rate + episode length")
    if k:
        out.append(f"    thresholds: stall<={k['thresholds']['max_stall_rate']} "
                   f"turns>={k['thresholds']['stall_turns']} "
                   f"ep_len<={k['thresholds']['max_ep_len_ratio']}x era "
                   f"({k['thresholds']['threshold_provenance']})")
        out.append(f"    {'step':>12}{'stall_rate':>12}{'mean_turns':>12}{'ep_bots':>9}"
                   f"{'ep_pool':>9}  verdict")
        for c in k["cycles"]:
            out.append(f"    {c['step']:>12,}{_f(c['stall_rate_captured']):>12}"
                       f"{_f(c['mean_turns_captured'], '.1f'):>12}"
                       f"{_f(c['ep_len_bots'], '.2f'):>9}{_f(c['ep_len_pool'], '.2f'):>9}  "
                       + ("KILL — " + "; ".join(c["breaches"]) if c["kill"] else "OK"))
        out.append(f"    stall rate source: {k['sources']['stall_rate']}")
    else:
        out.append("    not read")
    out.append("")

    mt = doc.get("untaught_meter")
    out.append("(4) UNTAUGHT METER — with a CONTINUATION control")
    if mt:
        out.append(f"    exit {mt['returncode']} in {mt['seconds']}s — {mt['control_note']}")
        for line in (mt.get("stdout") or "").rstrip().splitlines()[-40:]:
            out.append("    " + line)
        if mt["returncode"] != 0:
            for line in (mt.get("stderr") or "").rstrip().splitlines()[-20:]:
                out.append("    ! " + line)
    else:
        out.append("    skipped (--skip-meter)")
    out.append("")
    out.append("-" * w)
    out.append("NOT RUNNABLE HERE (§4.3 criteria this tool does not compute):")
    for gid, what, why in NOT_RUNNABLE:
        out.append(f"  {gid}: {what} — {why}")
    out.append("")
    out.append("FALSIFICATION CLAUSE (design §5.5, verbatim):")
    out.append("  " + FALSIFICATION_CLAUSE)
    return "\n".join(out)

