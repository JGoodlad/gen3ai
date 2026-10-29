"""THE CANARY FATAL RESUME DECISION — ``python -m main.ops.canary_decide <run> [--family c0|x47 | --bar B]``.

What it answers. A run launched with ``--compile-trainer`` stops at a canary FATAL (``agents.model.compile_canary``,
region R1 under the bf16 parity rule). The X35 staged plan's standing ruling is whether to RESUME that arm from its
newest checkpoint with ``--canary-shift-max 0.3`` or HOLD it for the orchestrator. This tool applies that rule to the
run's own records and prints one verdict; it resumes nothing and writes nothing under ``models/``.

THE RULE (orchestrator rulings 2026-10-20 ~03:15 / ~11:00, 2026-10-21 ~01:15 and ~01:55, 2026-10-24 ~01:40, from
Training Oct's X35 job log; this module replaces the job-directory ``canary_decide_v2.py``). RESUME iff ALL of:

  1. the LAST record of ``<run>/canary_verdicts.jsonl`` is a ``fatal`` (the ratio is judged ONLY at a FATAL: a PASS
     above the bar is descriptive, an UNCONFIRMED breach is not a FATAL);
  2. ZERO clip-branch switches in the last line of ``<run>/canary_disagreements.jsonl`` (``switched N row(s)``);
     a missing count is 0 ONLY for the eager-yardstick class (``bf16 EAGER micro-step has DRIFTED``: the compiled arm
     is not implicated), any other class with no count is a HOLD;
  3. for EVERY watched arm with a ``clip_margin`` record, compiled/eager ``max_abs_dlogp`` <= the BAR, with the eager
     shift > 0 and both fields present (a missing field is a HOLD);
  4. at least one arm was judged (a record that carries no ``clip_margin`` judges nothing, so it HOLDs; the job-
     directory version resumed on an empty judgement).

THE BARS APPLY ONLY TO RUNS ON THE GLOBAL MARGIN (ruling 2026-10-28, `designs/training/canary_recalibration_2026-10-28.md`
§6). A FATAL whose record carries a ``canary_rule_version`` other than ``global_m`` (`FAMILY_BAR_RULES`; a record with no
field was written at a pin before the per-row rule, ``agents.model.canary_row_bar``) was judged against each row's own
dither null, so the family ratio means nothing there: it HOLDs, whatever the bar, and the operator replays the dump
(``python -m main.ops.canary_dump_replay --dump``).

The same-precision gap |compiled - eager| is DESCRIPTIVE: it is bounded by the SUM of the two arms' shifts, so two
healthy bf16 arms drifting apart read up to 2x either (re-ruled 2026-10-21 ~01:15). It is printed, never judged.

THE BAR is the registered ratio bar of the arm FAMILY, fixed from the healthy-PASS maximum x 1.25 and not drifting:

  ``c0``  2.64  = 1.25 x 2.108 (C0 s3603 u500, the healthy PASS max over three seeds; ruling 2026-10-21 ~01:55)
  ``x47`` 2.40  = 1.25 x 1.893 (X47 s3601 u700, the healthy PASS max; ruling 2026-10-20 ~11:00)

The family is read from the run name (``FAMILY_NAME_PATTERNS``): the production arms ``rb_x35_{c0,a1*,sprt,lrfloor*,
d1wide,x52*,d3mut,n1p,x48kffn}_s*`` are all C0 (forks of P* at production pins; X48 by its PREREG's ruling,
``x48_kffn_production_2026-10-27/PREREG.md`` §7), X63-C's ``rb_x63c_s*`` is C0 (production C0's argv and
architecture, forking X26 @380M; ``x63c_control_replicate_2026-10-28/PREREG.md``), X63-U's ``rb_x63u_s*`` is C0 (the
same, + a team mix like N1-P's; ``x63_2026-10-29/PREREG.md``), the toy ``rb_x35t_c0_s*`` is C0 and
``rb_x35t_x47*_s*`` is X47; any other run (``rb_x35_r0_*``) needs ``--family`` or a typed ``--bar``. The bars'
provenance is in ``designs/training/compile_flags.md`` ("The canary-FATAL resume rule"). A typed ``--bar`` wins, is printed
with its source, and is appended to ``--log`` when given (the 2026-10-24 ruling that judged D1 s3505 at the C0
family's 2.64 was exactly that, typed once, this run only). A bar typed by hand into an untested script is where a
canary verdict goes wrong; this one is tested.

Exit 0 = RESUME, 3 = HOLD (every reason printed), 2 = usage / unreadable run. The resume itself is the operator's:
dry-run first, cite the ruling and the record path, keep every artifact (``designs/ops/TRAINING_RUN_SOP.md``).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

#: The registered family ratio bars (compiled / eager ``max_abs_dlogp`` at a FATAL). Provenance in the module doc.
FAMILY_BARS: Dict[str, float] = {"c0": 2.64, "x47": 2.40}

#: The canary rules the family bars judge (`agents.model.canary_row_bar.LEGACY_RULE_VERSION`: the compiled arm at the
#: global margin M). A record without ``canary_rule_version`` is ``global_m``; any other rule HOLDs (module doc).
FAMILY_BAR_RULES = ("global_m",)

#: The one disagreement class whose clip-switch count may be absent (see the module doc, rule 2).
EAGER_DRIFT = "bf16 EAGER micro-step has DRIFTED from the fp32"

RESUME, HOLD = "RESUME", "HOLD"
EXIT = {RESUME: 0, HOLD: 3}


#: Run name -> family, ANCHORED at the start of the name (a suffix such as ``.crashed-20261022`` is ignored). Every
#: production X35 arm is a fork of P* (rb_x26_s2601 @280M) at production pins, the same family as C0; the toy arms
#: are the toy C0 (bar C0) and the toy X47 (bar X47); X63-C (``rb_x63c_*``) forks X26 @380M at the production pin with
#: C0's argv and architecture, so it is C0's family too, and so is X63-U (``rb_x63u_*``: the same + a team mix, as
#: N1-P's). Anything else (``rb_x35_r0_*``, another batch) is unknown and HOLDs.
FAMILY_NAME_PATTERNS: "tuple[tuple[str, str], ...]" = (
    (r"rb_x35_(?:c0|a1[a-z0-9]*|sprt|lrfloor[a-z0-9]*|d1wide|x52[a-z0-9]*|d3mut|n1p|x48kffn)_s\d+", "c0"),
    (r"rb_x63c_s\d+", "c0"),
    (r"rb_x63u_s\d+", "c0"),
    (r"rb_x35t_c0_s\d+", "c0"),
    (r"rb_x35t_x47[a-z0-9]*_s\d+", "x47"),
)


def family_of(run_name: str) -> Optional[str]:
    """The arm family a run name belongs to (``c0`` / ``x47``), else ``None`` (an unknown name HOLDs)."""
    n = run_name.rstrip("/").split("/")[-1]
    for pattern, fam in FAMILY_NAME_PATTERNS:
        if re.match(pattern, n):
            return fam
    return None


@dataclass
class Decision:
    verdict: str
    run: str
    bar: Optional[float]
    bar_source: str
    update: Optional[int] = None
    reasons: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    arms: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    clip_switches: Optional[List[int]] = None

    def to_json(self) -> dict:
        return {"verdict": self.verdict, "run": self.run, "update": self.update, "bar": self.bar,
                "bar_source": self.bar_source, "reasons": self.reasons, "notes": self.notes,
                "arms": self.arms, "clip_switches": self.clip_switches}


def resolve_bar(run: str, family: Optional[str], bar: Optional[float]) -> "tuple[Optional[float], str]":
    """``(bar, source)``: a typed ``bar`` wins, then an explicit ``family``, then the run name's family."""
    if bar is not None:
        return float(bar), f"typed --bar {bar} (this run only)"
    fam = family or family_of(run)
    if fam is None:
        return None, f"no family bar for {run!r} (not a c0 / x47 run name; pass --family or --bar)"
    if fam not in FAMILY_BARS:
        raise ValueError(f"unknown family {fam!r}: registered {sorted(FAMILY_BARS)}")
    how = "--family" if family else "the run name"
    return FAMILY_BARS[fam], f"{fam} family bar {FAMILY_BARS[fam]} (registered; from {how})"


def decide(run: str, verdicts: Sequence[dict], disagreements: Sequence[str], *, bar: Optional[float],
           bar_source: str) -> Decision:
    """The pure rule. ``verdicts`` = the parsed ``canary_verdicts.jsonl``, ``disagreements`` = its raw lines."""
    d = Decision(verdict=HOLD, run=run, bar=bar, bar_source=bar_source)
    if not verdicts:
        d.reasons.append("canary_verdicts.jsonl holds no record")
        return d
    last = verdicts[-1]
    d.update = last.get("update")
    if str(last.get("verdict")).lower() != "fatal":
        d.reasons.append(f"the last verdict is {last.get('verdict')!r} at update {last.get('update')}, not fatal "
                         "(the ratio is judged only at a FATAL)")
        return d
    rule = str(last.get("canary_rule_version") or "global_m")
    if rule not in FAMILY_BAR_RULES:
        d.reasons.append(f"the FATAL was judged under the {rule!r} canary rule: the family bars apply only to "
                         f"{list(FAMILY_BAR_RULES)} runs (ruling 2026-10-28); replay the dump "
                         "(python -m main.ops.canary_dump_replay --dump)")
        return d
    if bar is None:
        d.reasons.append(bar_source)

    # (2) clip-branch switches
    if not disagreements:
        d.reasons.append("canary_disagreements.jsonl holds no line")
    else:
        text = disagreements[-1]
        sw = [int(x) for x in re.findall(r"switched (\d+) row", text)]
        if not sw and EAGER_DRIFT in text:
            sw = [0]
            d.notes.append("clip-switch count missing in the eager-yardstick class: read as 0")
        d.clip_switches = sw
        if not sw:
            d.reasons.append("clip-switch count missing")
        elif max(sw) != 0:
            d.reasons.append(f"clip branches switched: {sw}")

    # (3) the per-arm ratio
    for arm, rec in (last.get("watch") or {}).items():
        if not isinstance(rec, dict) or "clip_margin" not in rec:
            continue
        c = rec["clip_margin"] or {}
        m = c.get("max_abs_dlogp") or {}
        sp = (c.get("same_precision") or {}).get("max_abs_dlogp")
        comp, eag = m.get("compiled"), m.get("eager")
        d.arms[arm] = {"compiled": comp, "eager": eag, "same_precision": sp,
                       "ratio": (comp / eag) if (comp is not None and eag) else None}
        if comp is None or eag is None:
            d.reasons.append(f"{arm}: field missing (compiled {comp}, eager {eag})")
        elif eag <= 0:
            d.reasons.append(f"{arm}: eager shift {eag} is not positive")
        elif bar is not None and comp / eag > bar:
            d.reasons.append(f"{arm}: compiled/eager {comp}/{eag} = {comp / eag:.3f} > {bar} ({bar_source})")
    if not d.arms:
        d.reasons.append("no watched arm carries a clip_margin record: nothing was judged")
    if not d.reasons:
        d.verdict = RESUME
    return d


def _read_lines(path: Path) -> List[str]:
    return [ln for ln in path.read_text().splitlines() if ln.strip()]


def decide_run(run_dir: Path, *, family: Optional[str] = None, bar: Optional[float] = None) -> Decision:
    """Read ``run_dir``'s records and apply :func:`decide`. An unreadable record is a HOLD naming the file."""
    bar_v, source = resolve_bar(run_dir.name, family, bar)
    try:
        verdicts = [json.loads(ln) for ln in _read_lines(run_dir / "canary_verdicts.jsonl")]
    except (OSError, ValueError) as exc:
        return Decision(HOLD, run_dir.name, bar_v, source, reasons=[f"canary_verdicts.jsonl unreadable ({exc})"])
    try:
        dis = _read_lines(run_dir / "canary_disagreements.jsonl")
    except OSError:
        dis = []
    return decide(run_dir.name, verdicts, dis, bar=bar_v, bar_source=source)


def render(d: Decision) -> str:
    out = [f"{d.run}: last record update {d.update}; bar {d.bar} [{d.bar_source}]"]
    out += [f"  note: {n}" for n in d.notes]
    for arm, a in d.arms.items():
        ratio = "-" if a["ratio"] is None else f"{a['ratio']:.3f}"
        out.append(f"  {arm}: compiled {a['compiled']} eager {a['eager']} ratio {ratio} same-precision {a['same_precision']}")
    if d.verdict == RESUME:
        out.append(f"RESUME: update {d.update} FATAL is the bf16 tail on every judged arm (0 clip switches AND "
                   "compiled/eager <= the bar)")
    else:
        out.append("HOLD: " + "; ".join(d.reasons))
    return "\n".join(out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m main.ops.canary_decide", description=__doc__.split("\n\n")[0])
    p.add_argument("run", help="a run directory or a run NAME in the main checkout's models/")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--family", choices=sorted(FAMILY_BARS), help="the arm family (default: read from the run name)")
    g.add_argument("--bar", type=float, help="a typed ratio bar for THIS run only (printed, and logged with --log)")
    p.add_argument("--log", default=None, metavar="PATH", help="append the decision as one JSON line here")
    p.add_argument("--json", action="store_true", help="print the decision as JSON instead of text")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    from main.ops.run_ref import resolve_run_dir
    d = decide_run(resolve_run_dir(args.run), family=args.family, bar=args.bar)
    print(json.dumps(d.to_json(), indent=1) if args.json else render(d))
    if args.log:
        with open(args.log, "a") as fh:
            fh.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), **d.to_json()}) + "\n")
    return EXIT[d.verdict]


if __name__ == "__main__":
    sys.exit(main())
