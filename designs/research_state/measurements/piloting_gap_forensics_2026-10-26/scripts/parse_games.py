"""Parse banked anchor-read captures into one compact feature row per game (no play, no model).

Usage (run with the env interpreter, PYTHONPATH=<worktree>/src):
    python parse_games.py OUT.jsonl.gz

Each capture (``cells/<unit>/capture/<tag>.json``) holds the START command (both packed teams, the seed), every
CHOOSE command of both sides in submission order, and both sides' protocol chunks (each side's own view: its own HP
exact, the other's in percent). The focal side is our model (name ``Gen3AIAnchor*``); in X37a's ``kcal`` pair cell
it is the Kakuna side piloting held-out teams (``MetaKakunaHome*``:
the aliases are the peers' roles, not the team sets; checked by matching each side's six to the panel files).

Per game we keep, for BOTH sides (``f`` = focal, ``o`` = other):
- the six species (packed START team), the lead;
- every decision: turn, forced or not, action (``m:<moveid>`` / ``s:<species>``), the acting mon, the foe active,
  both actives' HP fractions before the decision;
- per move used: its protocol outcome flags (fail / immune / miss / supereffective / resisted / KO of the target);
- faints with turn and whether the last hit was direct (a foe's move) or residual;
- a material snapshot at every ``|turn|`` line: each side's summed HP fraction over six mons (an unrevealed mon
  counts 1.0: no damage can reach a mon that never entered) and mons alive.
The result is read from the protocol (``|win|`` / ``|tie|``) and cross-checked against ``games.jsonl``.
"""
from __future__ import annotations

import gzip
import json
import os
import re
import sys
from pathlib import Path

READS = Path("/home/goodlad/dev/gen3ai-reads")

# (read, cell prefix, opponent label, model label, our panel, their panel, focal name prefix)
CELLS = [
    ("x51", "pstar_home72", "Kakuna34", "P*", "home72", "home72", "Gen3AI"),
    ("x51", "pstar_hpfix", "Kakuna34", "P*", "H", "H", "Gen3AI"),
    ("x51", "pstar_forum72", "Kakuna34", "P*", "F", "F", "Gen3AI"),
    ("x51", "pstar_p72", "Kakuna34", "P*", "P72", "P72", "Gen3AI"),
    ("mm", "pm_home72", "PastaMittens", "P*", "home72", "home72", "Gen3AI"),
    ("mm", "pm_hpfix", "PastaMittens", "P*", "H", "H", "Gen3AI"),
    ("mm", "pm_forum72", "PastaMittens", "P*", "F", "F", "Gen3AI"),
    ("mm", "pm_p72", "PastaMittens", "P*", "P72", "P72", "Gen3AI"),
    ("x37a", "deploy_core", "Kakuna34", "X26@161M", "home72", "H", "Gen3AI"),
    ("x37a", "deploy_near", "Kakuna34", "X26@161M", "home72", "H", "Gen3AI"),
    ("x37a", "deploy_far", "Kakuna34", "X26@161M", "home72", "H", "Gen3AI"),
    ("x37a", "reverse", "Kakuna34", "X26@161M", "H", "home72", "Gen3AI"),
    ("x37a", "forum72", "Kakuna34", "X26@161M", "F", "F", "Gen3AI"),
    ("x37a", "p72", "Kakuna34", "X26@161M", "P72", "P72", "Gen3AI"),
    ("x37a", "kcal", "Kakuna34", "Kakuna34", "H", "home72", "MetaKakunaHome"),
]
READ_DIRS = {"x51": "x51_2026-10-20", "mm": "mm_strongest_2026-10-25", "x37a": "x37a_2026-10-18"}

HP_RE = re.compile(r"^(\d+)/(\d+)")


def to_id(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def hp_frac(cond: str) -> float:
    cond = cond.strip()
    if cond.startswith("0 fnt") or cond == "0":
        return 0.0
    m = HP_RE.match(cond)
    if not m:
        raise ValueError(f"unparsed HP {cond!r}")
    return int(m.group(1)) / int(m.group(2))


def packed_mons(team: str) -> list:
    """[[species_name, [move ids]], ...] from a packed team (species as written; normalised downstream)."""
    out = []
    for mon in team.split("]"):
        f = mon.split("|")
        out.append([f[1] or f[0], [to_id(x) for x in f[4].split(",") if x]])
    return out


class SideState:
    def __init__(self):
        self.nick2sp = {}
        self.hp = {}          # nick -> frac
        self.active = None    # nick

    def material(self):
        alive = sum(1 for v in self.hp.values() if v > 0) + (6 - len(self.hp))
        return round(sum(self.hp.values()) + (6 - len(self.hp)), 4), alive


def side_of(ident: str) -> tuple:
    # "p1a: Name" -> ("p1", "Name")
    pos, name = ident.split(": ", 1)
    return pos[:2], name


def parse_capture(path: Path, focal_prefix: str) -> dict:
    d = json.loads(path.read_text())
    start = json.loads(d["commands"][0][len("START "):])
    names = {"p1": start["p1"]["name"], "p2": start["p2"]["name"]}
    fs = "p1" if names["p1"].startswith(focal_prefix) else "p2"
    if not names[fs].startswith(focal_prefix):
        raise ValueError(f"no focal side in {path}")
    osd = "p2" if fs == "p1" else "p1"
    teams = {s: packed_mons(start[s]["team"]) for s in ("p1", "p2")}
    cmds = {"p1": [], "p2": []}
    for c in d["commands"][1:]:
        parts = c.split(" ", 3)
        if parts[0] != "CHOOSE":
            continue
        cmds[parts[1]].append(parts[2] + ":" + (to_id(parts[3]) if len(parts) > 3 else ""))

    st = {"p1": SideState(), "p2": SideState()}
    turn = 0
    snaps = []
    faints = []
    moves = []        # (turn, side, user_sp, move_id, target_sp, flags)
    decisions = {"p1": [], "p2": []}
    req_i = {"p1": 0, "p2": 0}
    result = None
    last_move = None   # index into moves of the move whose effects are being read
    last_hit_direct = {}  # (side,nick) -> True if last damage line had no [from]
    # walk the focal side's view for state; requests from each side's own stream
    for side, text in d["chunks"]:
        for line in text.split("\n"):
            if line.startswith("|request|"):
                r = json.loads(line[len("|request|"):])
                if r.get("wait") or r.get("teamPreview"):
                    continue
                forced = bool(r.get("forceSwitch"))
                k = req_i[side]
                req_i[side] += 1
                act = cmds[side][k] if k < len(cmds[side]) else None
                me, foe = st[side], st["p2" if side == "p1" else "p1"]
                decisions[side].append({
                    "t": turn, "forced": forced, "a": act,
                    "me": me.nick2sp.get(me.active) if me.active else None,
                    "foe": foe.nick2sp.get(foe.active) if foe.active else None,
                    "me_hp": round(me.hp.get(me.active, 1.0), 3) if me.active else None,
                    "foe_hp": round(foe.hp.get(foe.active, 1.0), 3) if foe.active else None,
                    "n_sw": sum(1 for p in r["side"]["pokemon"]
                                if not p.get("active") and not p["condition"].startswith("0 fnt")),
                })
                continue
            if side != fs:
                continue
            p = line.split("|")
            if len(p) < 2:
                continue
            tag = p[1]
            if tag in ("switch", "drag"):
                s, nick = side_of(p[2])
                sp = to_id(p[3].split(",")[0])
                ss = st[s]
                ss.nick2sp[nick] = sp
                ss.active = nick
                ss.hp[nick] = hp_frac(p[4])
                last_move = None
            elif tag in ("-damage", "-heal", "-sethp"):
                s, nick = side_of(p[2])
                frac = hp_frac(p[3])
                st[s].hp[nick] = frac
                direct = not any(x.startswith("[from]") for x in p[4:])
                last_hit_direct[(s, nick)] = direct
                if direct and last_move is not None and tag == "-damage" and frac == 0.0:
                    moves[last_move][5]["ko"] = 1
            elif tag == "faint":
                s, nick = side_of(p[2])
                st[s].hp[nick] = 0.0
                faints.append({"t": turn, "side": "f" if s == fs else "o",
                               "sp": st[s].nick2sp.get(nick), "direct": last_hit_direct.get((s, nick), False)})
            elif tag == "move":
                s, nick = side_of(p[2])
                tgt = None
                if len(p) > 4 and p[4].startswith("p"):
                    ts, tn = side_of(p[4])
                    tgt = st[ts].nick2sp.get(tn)
                moves.append((turn, "f" if s == fs else "o", st[s].nick2sp.get(nick), to_id(p[3]), tgt, {}))
                last_move = len(moves) - 1
            elif tag in ("-fail", "-immune", "-miss", "-supereffective", "-resisted", "-crit") and last_move is not None:
                moves[last_move][5][tag[1:]] = 1
            elif tag == "-activate" and last_move is not None and len(p) > 3 and "Protect" in p[3]:
                moves[last_move][5]["protected"] = 1
            elif tag == "turn":
                turn = int(p[2])
                mf, af = st[fs].material()
                mo, ao = st[osd].material()
                snaps.append((turn, mf, af, mo, ao))
                last_move = None
            elif tag == "upkeep":
                last_move = None
            elif tag == "win":
                result = 1.0 if p[2] == names[fs] else 0.0
            elif tag == "tie":
                result = 0.5
    mf, af = st[fs].material()
    mo, ao = st[osd].material()
    final = (turn, mf, af, mo, ao)
    for s in ("p1", "p2"):
        # a game stopped by the 250-turn forfeit leaves each side's last request unanswered
        if req_i[s] == len(cmds[s]) + 1 and turn >= 250:
            decisions[s].pop()
        elif req_i[s] != len(cmds[s]):
            raise ValueError(f"{path}: side {s} has {req_i[s]} decisions but {len(cmds[s])} commands")
    return {
        "tag": d["tag"], "fs": fs, "result": result, "turns": turn,
        "f_team": teams[fs], "o_team": teams[osd],
        "f_dec": decisions[fs], "o_dec": decisions[osd],
        "moves": moves, "faints": faints, "snaps": snaps, "final": final,
    }


def main(out: str) -> None:
    n = 0
    bad = 0
    with gzip.open(out, "wt") as fo:
        for read, prefix, opp, model, our_p, their_p, focal in CELLS:
            cells_dir = READS / READ_DIRS[read] / "cells"
            for unit in sorted(os.listdir(cells_dir)):
                m = re.fullmatch(re.escape(prefix) + r"_(\d+)", unit)
                if not m:
                    continue
                ud = cells_dir / unit
                rows = {}
                gj = ud / "games.jsonl"
                if gj.exists():
                    for line in gj.read_text().splitlines():
                        r = json.loads(line)
                        rows[r["battle_tag"]] = r
                for cap in sorted((ud / "capture").glob("*.json")):
                    try:
                        g = parse_capture(cap, focal)
                    except Exception as e:  # noqa: BLE001 — every skip is counted and reported
                        bad += 1
                        print(f"SKIP {cap}: {e}", file=sys.stderr)
                        continue
                    row = rows.get(g["tag"])
                    if row is not None:
                        rw = 0.5 if row.get("result") == "tie" else (1.0 if row["won"] else 0.0)
                        if g["result"] is None:
                            g["result"] = rw
                        elif g["result"] != rw:
                            raise SystemExit(f"result mismatch {cap}: protocol {g['result']} row {rw}")
                        g["half"], g["index"] = row["half"], row["index"]
                        g["our_team_file"] = row.get("our_team")
                        g["forfeit"] = bool(row.get("hit_forfeit_limit"))
                    elif g["result"] is None:
                        bad += 1
                        print(f"SKIP {cap}: no result", file=sys.stderr)
                        continue
                    g.update(read=read, cell=prefix, unit=unit, seed=int(m.group(1)), opp=opp, model=model,
                             our_panel=our_p, their_panel=their_p)
                    fo.write(json.dumps(g, separators=(",", ":")) + "\n")
                    n += 1
    print(f"wrote {n} games, skipped {bad}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1])
