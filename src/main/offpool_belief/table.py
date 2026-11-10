"""The meter's TABLE: score files in, one block per panel out — every arm's belief against the Smogon prior, and
(``versus``) every arm against a named reference arm, all team-clustered and paired on the bank's teams.

An ARM is one or more checkpoints (seeds) scored on the SAME banks; its per-team value is the mean over its
checkpoints of each checkpoint's team mean, so seeds are averaged inside a team and the interval resamples TEAMS
only. With more than one checkpoint the block also carries the per-checkpoint range of the panel mean: the seed
spread the team interval does not contain.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from main.offpool_belief import OffPoolBeliefError
from main.offpool_belief import metrics as M
from main.offpool_belief import stats as S
from main.offpool_belief.score import arm_team_metric, team_metric

TABLE_SCHEMA = "gen3_offpool_belief_table_v1"
#: The table's rows, in order, with what each one is called on the page.
METRIC_TITLES = {
    "species_logp": "hidden species: mean log presence of the true species",
    "species_hit": "hidden species: share of the true species in the top k",
    "moves_logp": "unshown moves of a revealed mon: mean log probability",
    "moves_prec": "unshown moves of a revealed mon: precision at |unshown|",
    "items_logp": "unstated item of a revealed mon: log probability",
    "items_top1": "unstated item of a revealed mon: top-1 rate",
}


def _panel_blocks(arm: str, docs: Sequence[Mapping[str, Any]], panel: str) -> List[Mapping[str, Any]]:
    missing = [d["label"] for d in docs if panel not in d["panels"]]
    if missing:
        raise OffPoolBeliefError(f"arm {arm!r}: {missing} carry no score on panel {panel!r}")
    return [d["panels"][panel] for d in docs]


def build(arms: Mapping[str, Sequence[Mapping[str, Any]]], *, versus: Optional[str] = None,
          panels: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """``arms``: name -> its score documents. Returns the table document (:data:`TABLE_SCHEMA`)."""
    if not arms:
        raise OffPoolBeliefError("a table needs at least one arm")
    if versus is not None and versus not in arms:
        raise OffPoolBeliefError(f"--versus {versus!r} is not one of the arms {sorted(arms)}")
    first = next(iter(arms.values()))
    names = list(panels) if panels else list(first[0]["panels"])
    out: Dict[str, Any] = {"schema": TABLE_SCHEMA, "versus": versus, "panels": {},
                           "arms": {a: [{"label": d["label"], "checkpoint": d["checkpoint"]} for d in docs]
                                    for a, docs in arms.items()},
                           "bootstrap": {"draws": S.DEFAULT_DRAWS, "seed": S.DEFAULT_SEED, "unit": "team"}}
    for panel in names:
        blocks = {a: _panel_blocks(a, docs, panel) for a, docs in arms.items()}
        ref = next(iter(blocks.values()))[0]
        if len({b["bank"]["rows_sha256"] for bs in blocks.values() for b in bs}) != 1:
            raise OffPoolBeliefError(f"panel {panel!r}: the arms were scored on different banks (rows differ); a "
                                     "table compares arms on ONE bank")
        n_teams = len(ref["team_labels"])
        idx = S.boot_index(n_teams)
        cell: Dict[str, Any] = {"kind": ref["kind"], "held_out": ref["held_out"], "n_teams": n_teams,
                                "bank": ref["bank"], "metrics": {}, "obs_gap": {}, "excluded": {}}
        for a, bs in blocks.items():
            cell["obs_gap"][a] = sorted({tuple(b["obs_gap"]["rows_between"] or ()) for b in bs})
            cell["excluded"][a] = {k: int(np.sum(v)) for k, v in bs[0]["excluded"].items()}
        for metric in M.METRICS:
            row: Dict[str, Any] = {"units": int(np.sum(ref["units"][M.FAMILY_OF[metric]])), "arms": {}}
            base = arm_team_metric(blocks[versus], metric, M.POST) if versus is not None else None
            for a, bs in blocks.items():
                post = arm_team_metric(bs, metric, M.POST)
                prior = arm_team_metric(bs, metric, M.PRIOR)
                per_ckpt = [float(np.nanmean(team_metric(b, metric, M.POST))) for b in bs]
                arm_cell: Dict[str, Any] = {
                    "n_checkpoints": len(bs), "post": S.mean_ci(post, idx), "prior": S.mean_ci(prior, idx),
                    "post_minus_prior": S.paired_diff(post, prior, idx),
                    "checkpoint_range": [min(per_ckpt), max(per_ckpt)],
                }
                for v in bs[0]["variants"]:
                    if v not in (M.POST, M.PRIOR) and all(v in b["variants"] for b in bs):
                        arm_cell[f"{v}_minus_prior"] = S.paired_diff(arm_team_metric(bs, metric, v), prior, idx)
                if base is not None and a != versus:
                    arm_cell[f"minus_{versus}"] = S.paired_diff(post, base, idx)
                row["arms"][a] = arm_cell
            cell["metrics"][metric] = row
        out["panels"][panel] = cell
    return out


def _ci(c: Mapping[str, Any], signed: bool = False) -> str:
    if c["mean"] is None:
        return "n/a"
    fmt = "{:+.3f}" if signed else "{:.3f}"
    if c["lo"] is None:
        return fmt.format(c["mean"])
    return f"{fmt.format(c['mean'])} [{fmt.format(c['lo'])}, {fmt.format(c['hi'])}]"


def verdict(c: Mapping[str, Any]) -> str:
    """How a paired difference reads: its 95 % team interval is wholly above zero (BETTER), wholly below (WORSE),
    or straddles it (NOT DETECTED). An interval that touches zero exactly is NOT DETECTED."""
    if c["mean"] is None or c["lo"] is None:
        return "n/a"
    if c["lo"] > 0.0:
        return "BETTER"
    if c["hi"] < 0.0:
        return "WORSE"
    return "NOT DETECTED"


def markdown(doc: Mapping[str, Any]) -> str:
    """The table as markdown. Every number is a mean over teams of team means; brackets are 95 % team-bootstrap
    intervals; a Δ column's word is :func:`verdict`."""
    versus = doc["versus"]
    lines: List[str] = []
    for panel, cell in doc["panels"].items():
        b = cell["bank"]
        lines += [(f"### {panel} — {cell['kind']} ({cell['n_teams']} teams, {b['n_games']} games, {b['n_rows']} "
                   f"rows, bank {b['rows_sha256'][:12]})"), ""]
        for metric, row in cell["metrics"].items():
            mixes = sorted({k for a in row["arms"].values() for k in a if k.startswith("mix")})
            head = ["arm", "ckpts", "belief", "Smogon prior", "belief − prior", ""]
            head += [f"{m.replace('_minus_prior', '')} − prior" for m in mixes]
            if versus is not None:
                head += [f"belief − {versus}", ""]
            lines += [f"**{metric}** ({METRIC_TITLES[metric]}; {row['units']} units, higher is better)", "",
                      "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
            for a, c in row["arms"].items():
                r = [a, str(c["n_checkpoints"]), _ci(c["post"]), _ci(c["prior"]), _ci(c["post_minus_prior"], True),
                     verdict(c["post_minus_prior"])]
                r += [_ci(c[m], True) if m in c else "n/a" for m in mixes]
                if versus is not None:
                    d = c.get(f"minus_{versus}")
                    r += [_ci(d, True), verdict(d)] if d is not None else ["(reference)", ""]
                lines.append("| " + " | ".join(r) + " |")
            lines.append("")
        gaps = {a: g for a, g in cell["obs_gap"].items() if any(g)}
        if gaps:
            lines += ["Observation-semantics rows between a checkpoint's recorded config and the bank's encoding "
                      "(scored through them, never trained on them): "
                      + "; ".join(f"{a}: {[list(x) for x in g]}" for a, g in gaps.items()), ""]
        exc = next(iter(cell["excluded"].values()))
        if exc:
            lines += ["Units not scored / flagged (first arm): " + ", ".join(f"{k} {v}" for k, v in exc.items()), ""]
    return "\n".join(lines).rstrip() + "\n"
