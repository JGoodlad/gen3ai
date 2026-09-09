"""Fold the tmp readouts into the SMALL committed artifacts + print README-ready tables.

Only summary JSON is committed — never the prediction columns and never the 128-dim feature
tensor. `run.sh` regenerates everything from `models/` (plus the probe read's frozen forward) in
about an hour.

THE READING RULE IS APPLIED HERE, IN CODE, so the verdict per substrate is a function of the
numbers rather than of the author. The three outcomes were registered before the fit:

  1. (a) RECOVERS  — the terminal-target refit closes the between-opponent spread ratio toward 1
                     and its prediction decodes the opponent's class far above the online head's.
                     => the ONLINE OPTIMISATION is the defect; the treatment is how the head is
                        TRAINED (a value replay, a periodic head refit, a head-specific lr).
  2. (a) FAILS and (b) RECOVERS — the same features and the same head reach the separation once
                     the per-episode label variance is removed.
                     => TARGET-VARIANCE levers (MC / counterfactual labels, search leaves,
                        outcome balancing).
  3. BOTH FAIL while `value_pooled` still linearly decodes the opponent
                     => the MAPPING is what the head lacks: capacity / the LayerNorm-128
                        bottleneck / a separate value trunk.

and one guard that the registration did not name but the data can force:

  0. THE CEILING FAILS — `cond_oracle` (the conditional target emitted verbatim as a prediction)
     itself does not reach a ratio near 1. Then (b)'s failure says nothing about the head, because
     the label it was handed did not contain the separation either, and outcome 2 is untestable on
     this target. This is checked FIRST and reported as an INCONCLUSIVE (b), never folded into 3.

A condition RECOVERS only when BOTH of these hold, each on the delta's own battle-clustered CI:
  * its turn-1–3 spread ratio improves on the online head with the delta's CI clear of zero, AND
    the ratio's own CI lower bound clears `RATIO_FLOOR`;
  * its prediction's opponent-CLASS decode at turn 1 clears its own permutation null AND improves
    on the online head's with the delta's CI clear of zero.
A condition that satisfies the first pair but not the ratio floor is PARTIAL — it moved the meter
without closing it, which is a different treatment recommendation from either 1 or 2.
"""
from __future__ import annotations

import argparse
import json
import os

CONDITIONS = ("online", "online_rec", "lin_term", "mlp_term", "mlp_term_ft",
              "lin_cond", "mlp_cond", "mlp_cond_ft", "cond_oracle")
LABEL = {"online": "online head, FROZEN forward (c)",
         "online_rec": "online head, RECORDED per-cycle (c-conservative)", "lin_term": "linear · terminal (a-floor)",
         "mlp_term": "MLP · terminal, scratch (a)", "mlp_term_ft": "MLP · terminal, fine-tuned (a-ft)",
         "lin_cond": "linear · conditional (b-floor)", "mlp_cond": "MLP · conditional, scratch (b)",
         "mlp_cond_ft": "MLP · conditional, fine-tuned (b-ft)",
         "cond_oracle": "the conditional TARGET itself (ceiling)"}
RATIO_FLOOR = 0.80          # "the identity is ~1 for a head that conditions"


def _f(v, n=3):
    return "—" if v is None else f"{v:.{n}f}" if isinstance(v, float) else str(v)


def _ci(v, n=3):
    if not v or v[0] is None:
        return "—"
    return f"[{v[0]:.{n}f}, {v[1]:.{n}f}]"


def clear_of_zero(ci, sign=+1):
    if not ci or ci[0] is None:
        return False
    return ci[0] > 0 if sign > 0 else ci[1] < 0


def decode_detected(cell, k):
    lo = (cell.get("ci") or {}).get(k, [None, None])[0]
    p95 = (cell.get("null_p95") or {}).get(k)
    return bool(lo is not None and p95 is not None and lo > p95)


def status(read, c):
    """RECOVERS / PARTIAL / FAILS for one condition, by the registered rule."""
    sp = read["spread"]
    d = sp["delta"].get(f"{c}-online|t1_3|ratio", {})
    ratio_ci = sp["ci"].get(f"{c}|t1_3|ratio", [None, None])
    cell = (read.get("decode") or {}).get("t1|opp_class")
    dec_ok = False
    if cell:
        dec_ok = decode_detected(cell, c) and clear_of_zero(
            (cell.get("delta_ci") or {}).get(f"{c}-online", [None, None]))
    moved = clear_of_zero(d.get("ci"))
    closed = ratio_ci[0] is not None and ratio_ci[0] > RATIO_FLOOR
    if moved and closed and dec_ok:
        return "RECOVERS"
    if moved or dec_ok:
        return "PARTIAL"
    return "FAILS"


def verdict(read):
    st = {c: status(read, c) for c in CONDITIONS if c != "online"}
    ceil = read["spread"]["ci"].get("cond_oracle|t1_3|ratio", [None, None])
    ceiling_ok = ceil[0] is not None and ceil[0] > RATIO_FLOOR
    a = "RECOVERS" if "RECOVERS" in (st["mlp_term"], st["lin_term"]) else \
        ("PARTIAL" if "PARTIAL" in (st["mlp_term"], st["lin_term"]) else "FAILS")
    b = "RECOVERS" if "RECOVERS" in (st["mlp_cond"], st["mlp_cond_ft"], st["lin_cond"]) else \
        ("PARTIAL" if "PARTIAL" in (st["mlp_cond"], st["mlp_cond_ft"], st["lin_cond"]) else "FAILS")
    if a == "RECOVERS":
        v = "1 — THE ONLINE OPTIMISATION IS THE DEFECT"
    elif not ceiling_ok:
        v = ("(b) INCONCLUSIVE — the conditional TARGET's own ceiling does not reach the identity, "
             f"so its failure is not evidence about the head (ceiling ratio CI {_ci(ceil)}). "
             + ("(a) PARTIAL." if a == "PARTIAL" else "(a) FAILS."))
    elif b == "RECOVERS":
        v = "2 — THE PER-EPISODE LABEL VARIANCE IS THE BINDING CONSTRAINT"
    elif a == "FAILS" and b == "FAILS":
        v = "3 — THE MAPPING (head capacity / the LayerNorm-128 bottleneck) IS WHAT IS MISSING"
    else:
        v = (f"MIXED — (a) {a}, (b) {b}; read the table. A condition that MOVED the meter without "
             f"CLOSING it is a partial recovery and its own treatment row.")
    return {"reading": v, "per_condition": st, "ceiling_reaches_identity": ceiling_ok,
            "ceiling_ratio_ci": ceil, "a": a, "b": b}


# ──────────────────────────────────────────────────────────────────────────────
def spread_table(read, title):
    sp = read["spread"]
    out = [f"\n### {title} — the mixture identity (between-opponent spread ratio)\n",
           "| condition | turn 1 | turns 1–3 | all states | Δ(vs online) @t1–3 | bias slope /100 Elo |",
           "|---|---|---|---|---|---|"]
    for c in CONDITIONS:
        r = [f"{_f(sp['point'].get(f'{c}|{b}|ratio'))} {_ci(sp['ci'].get(f'{c}|{b}|ratio'))}"
             for b in ("t1", "t1_3", "all")]
        d = sp["delta"].get(f"{c}-online|t1_3|ratio", {})
        dtxt = "—" if c == "online" else (
            ("**" if clear_of_zero(d.get("ci")) else "")
            + f"{_f(d.get('point'))} {_ci(d.get('ci'))}"
            + ("**" if clear_of_zero(d.get("ci")) else ""))
        sl = f"{_f(sp['point'].get(f'{c}|slope'), 4)} {_ci(sp['ci'].get(f'{c}|slope'), 4)}"
        out.append(f"| {LABEL[c]} | {r[0]} | {r[1]} | {r[2]} | {dtxt} | {sl} |")
    return out


def decode_table(read, title):
    out = [f"\n### {title} — what the PREDICTION decodes (the probe read's rows)\n"]
    for key in ("t1|opp_class", "t1_3|opp_class", "t1|own_team_wr", "t1_3|own_team_wr"):
        cell = (read.get("decode") or {}).get(key)
        if not cell:
            continue
        bk, tg = key.split("|")
        out += [f"\n**{tg}** at {'turn 1' if bk == 't1' else 'turns 1–3'} "
                f"({cell['task'].upper()}, n = {cell['n_states']} states / {cell['n_battles']} battles)\n",
                "| decoder | score | 95 % CI | null p95 | Δ vs online | Δ vs value_pooled |",
                "|---|---|---|---|---|---|"]
        for c in ("pooled",) + CONDITIONS:
            s = cell["score"].get(c)
            mark = "**" if decode_detected(cell, c) else ""
            dvo = (cell.get("delta_ci") or {}).get(f"{c}-online")
            dvp = (cell.get("delta_ci") or {}).get(f"{c}-pooled")
            name = "`value_pooled` (the head's input)" if c == "pooled" else LABEL[c]
            out.append(f"| {name} | {mark}{_f(s)}{mark} | {_ci(cell['ci'].get(c))} | "
                       f"{_f(cell['null_p95'].get(c))} | {_ci(dvo) if c != 'online' else '—'} | "
                       f"{_ci(dvp)} |")
    return out


def brier_table(read, title, mask="all"):
    bl = read["brier"][mask]
    out = [f"\n### {title} — Brier on HELD-OUT battles ({'all states' if mask == 'all' else 'turns 1–3'})\n",
           "| condition | Brier (held out) | 95 % CI | Δ vs online | reliability | resolution | "
           "uncertainty | skill | Brier (in fold) |", "|---|---|---|---|---|---|---|---|---|"]
    for c in CONDITIONS:
        h = bl["held_out"][c]
        t = bl["in_fold"].get(c) or {}
        d = bl["brier_delta_ci"].get(f"{c}-online")
        mark = "**" if (d and d[1] is not None and d[1] < 0) else ""
        out.append(f"| {LABEL[c]} | {mark}{_f(h['brier'], 4)}{mark} | {_ci(bl['brier_ci'][c], 4)} | "
                   f"{_ci(d, 4) if c != 'online' else '—'} | {_f(h['reliability'], 4)} | "
                   f"{_f(h['resolution'], 4)} | {_f(h['uncertainty'], 4)} | {_f(h['skill'], 3)} | "
                   f"{_f(t.get('brier'), 4)} |")
    return out


def pair_table(read, title):
    sp = read["spread"]
    keys = [("online-online_rec", "frozen forward − recorded (the comparator's own gap)"),
            ("mlp_term-online_rec", "(a) − the CONSERVATIVE online row"),
            ("mlp_cond-online_rec", "(b) − the CONSERVATIVE online row"),
            ("mlp_term-mlp_term_ft", "scratch − fine-tuned, TERMINAL (the ORDER hazard)"),
            ("mlp_cond-mlp_cond_ft", "scratch − fine-tuned, CONDITIONAL"),
            ("mlp_cond-mlp_term", "conditional − terminal, same head (the TARGET's own effect)"),
            ("mlp_term-lin_term", "MLP − linear, terminal (CAPACITY at fixed target)"),
            ("mlp_cond-lin_cond", "MLP − linear, conditional"),
            ("cond_oracle-mlp_cond", "the ceiling − the fit that chased it")]
    out = [f"\n### {title} — the isolating contrasts, each on the delta's own CI\n",
           "| contrast | Δ ratio @turn 1 | Δ ratio @turns 1–3 | Δ ratio @all | Δ Brier (all states) |",
           "|---|---|---|---|---|"]
    for k, lab in keys:
        cells = []
        for b in ("t1", "t1_3", "all"):
            d = sp["delta"].get(f"{k}|{b}|ratio", {})
            m = "**" if (clear_of_zero(d.get("ci")) or clear_of_zero(d.get("ci"), -1)) else ""
            cells.append(f"{m}{_f(d.get('point'))} {_ci(d.get('ci'))}{m}")
        db = read["brier"]["all"]["brier_delta_ci"].get(k)
        out.append(f"| {lab} | {cells[0]} | {cells[1]} | {cells[2]} | {_ci(db, 4)} |")
    return out


def main(a):
    reads = {}
    for tag, fn in (("A", "read_A.json"), ("CTRL", "read_CTRL.json"),
                    ("A_train", "read_A_train.json"), ("A_raw", "read_A_raw.json"),
                    ("A_cell", "read_A_cell.json")):
        p = os.path.join(a.tmp, fn)
        if os.path.exists(p):
            reads[tag] = json.load(open(p))
    if "A" not in reads:
        raise SystemExit("REFUSED: read_A.json is missing — nothing to summarise.")

    stats = {"substrates": {}, "verdict": {}, "counter_hypotheses": {}}
    for tag in ("A", "CTRL"):
        if tag in reads:
            stats["substrates"][tag] = reads[tag]
            stats["verdict"][tag] = verdict(reads[tag])
    for tag in ("A_train", "A_raw", "A_cell"):
        if tag in reads:
            r = reads[tag]
            stats["counter_hypotheses"][tag] = {
                "n_states": r["n_states"], "n_battles": r["n_battles"],
                "train_columns": r["train_columns"],
                "conditional_target": r["conditional_target"],
                "ratio_t1_3": {c: r["spread"]["point"].get(f"{c}|t1_3|ratio") for c in CONDITIONS},
                "ratio_t1_3_ci": {c: r["spread"]["ci"].get(f"{c}|t1_3|ratio") for c in CONDITIONS},
                "delta_vs_online_t1_3": {c: r["spread"]["delta"].get(f"{c}-online|t1_3|ratio")
                                         for c in CONDITIONS if c != "online"},
                "brier_all": {c: r["brier"]["all"]["held_out"][c]["brier"] for c in CONDITIONS},
                "slope": {c: r["spread"]["point"].get(f"{c}|slope") for c in CONDITIONS}}
    with open(os.path.join(a.out_dir, "refit_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)

    md = ["# Head-refit tables — every cell, every CI",
          "",
          "Generated by `summarize.py`; the reading in `README.md` is this file plus the "
          "counter-hypothesis runs. **Bold** = the delta's own 95 % CI is clear of zero (for a "
          "decode, additionally that the score clears its own permutation null).", ""]
    for tag, title in (("A", "Arm A @74M"), ("CTRL", "CTRL @10M")):
        if tag not in reads:
            continue
        r = reads[tag]
        md += [f"\n## {title} — {r['n_states']} states / {r['n_battles']} battles\n",
               f"**Reading: {verdict(r)['reading']}**", ""]
        md += spread_table(r, title)
        md += pair_table(r, title)
        md += brier_table(r, title, "all")
        md += brier_table(r, title, "t1_3")
        md += decode_table(r, title)
    for tag, title in (("A_train", "Arm A — IN-FOLD columns (the overfitting counter-hypothesis)"),
                       ("A_raw", "Arm A — UNWEIGHTED (the HT-weight counter-hypothesis)"),
                       ("A_cell", "Arm A — raw per-(opponent, team) LOO cell target "
                                  "(the leakage counter-hypothesis)")):
        if tag not in reads:
            continue
        r = reads[tag]
        md += [f"\n## {title}\n"]
        md += spread_table(r, title)
        md += brier_table(r, title, "all")
    with open(os.path.join(a.out_dir, "tables.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    for tag in stats["verdict"]:
        print(f"{tag}: {stats['verdict'][tag]['reading']}")
        print("   ", stats["verdict"][tag]["per_condition"])
    print("wrote refit_stats.json + tables.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmp", required=True)
    ap.add_argument("--out-dir", default=".")
    main(ap.parse_args())
