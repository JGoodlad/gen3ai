"""Q19 searched-cost read, STAGE 0 (the offline screen, OUTCOME-BLIND): every candidate preset on a registered,
stratified sample of S'-HONEST's banked searched battles, each battle's whole decision sequence from turn 10 rebuilt
from its ``capture/`` input log and searched by ONE live searcher (``../DRAFT_PREREG.md`` §3, §7 R6; stage 1b's
``../../sprime_honest_inc_stage1b_2026-11-06/seq_probe.py`` generalised to any preset and to the captures).

NEVER READ: ``games.jsonl``, ``summary.json``, a referee field, any game result. A cell is read only through its
``search_decisions.jsonl`` (the banked decision records: cost, picks, identity digests) and its ``capture/battle-*.json``
(``{tag, seed, commands, chunks}``: the input the sim took and each side's stream).

THE HONEST LOG, rebuilt from a capture (:func:`reconstruct`, :func:`cut`): ``search_side.honest_log`` reads the front
end's ``choice_log`` (``utils/bridge/ws_frontend.py``: each ``CHOOSE`` / ``FORCELOSE`` written to the child, tagged with
the writing side's request update ``req_n``; an ``[Invalid choice]`` deletes that side's last entry). The capture keeps
the SIM WRITE ORDER of the commands and each side's chunk stream but not their interleaving, so the update of each
command is re-derived per side from its own stream, exactly as the front end counts: a non-empty ``|request|`` bumps
``req_n`` unless an ``[Unavailable choice]`` came before it (a re-request of the SAME update); a non-``wait`` request
(or a re-request, or an ``[Invalid choice]``) opens one choice, which the side's next command answers; an ``[Invalid
choice]`` drops the command just answered. Our decision i is our i-th written ``CHOOSE`` (a dropped one included: the
live searcher decided it); its cut holds every surviving entry of an update before ours and our own earlier entries of
our update, ordered (update, p1 before p2, write order), each switch re-spelled by ``search_side.core_command``.
HARD CHECKS per battle (any failure is a named ``group_error`` row, never a silent score): our decisions number the
banked records; every search-path decision's cut hashes (``forced_branch.log_sha256``) to its banked ``log_sha256``,
its length is the banked ``at``, its update the banked ``update``, its key (``search_side.deal_key`` of the START)
the banked ``key``; ``honest_inv.sequence`` on the last one (the searcher's engine checking every cut's root) yields
exactly the banked search-path (turn, n) set, and gives each its mask and tokens; ``sign_gate.GateScorer.root`` then
checks each cut's replay against the banked ``frame_sha256``. ``pi_argmax`` is the banked one.

    PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3; export PYTHONPATH=src
    $PY screen.py sample --cells CELLS --salt S --per-panel 120 --out sample.json
    $PY screen.py run --preset P@10 --model ZIP --model-sha256 H --sample sample.json --out DIR [--salt S]
        [--shard i/k] [--max-battles N]
    $PY screen.py summarize --arm DIR [--arm DIR ...] --ref DIR [--floor DIR] --cpu-meter M --out summary.json
    $PY screen.py selfcheck
"""
import argparse
import gzip
import hashlib
import json
import os
import re
import sys
import time
from fractions import Fraction
from pathlib import Path

PANELS = {"heldout72_hpfix": "H", "forum72": "F", "p72": "P72"}
PAT = re.compile(r"^pstar_s__(forum72|heldout72_hpfix|p72)__(\d+)__g(\d+)$")
FIELDS = ("searched", "why", "error", "played", "deviation", "cpu_s", "world_cpu_s", "playout_cpu_s", "wall_s", "q",
          "log_pi", "root_v", "spent", "world_tries", "search_key")
BANKED = ("searched", "why", "played", "search_key", "wall_s", "cpu_s")
INVALID = "|error|[Invalid choice]"           # ws_frontend._INVALID_CHOICE_PREFIX
UNAVAILABLE = "|error|[Unavailable choice]"   # ws_frontend._UNAVAILABLE_CHOICE_PREFIX
#: rule 8's band: a value within this of its bar is ``near`` and does NOT pass
NEAR = Fraction(5, 10000)


class ReconError(ValueError):
    """A capture that cannot be read back into the battle's choice log, or one that disagrees with the banked
    records: the battle is a named ``group_error``."""


# ------------------------------------------------------------------------------------------ the capture -> the log
def _tag_side(side, cmds, chunks):
    """Each of ``side``'s commands gets its update (``upd``) and ``dropped`` from the side's own chunk stream
    (:func:`reconstruct`)."""
    mine = [e for e in cmds if e["cmd"].startswith("CHOOSE ")]
    forcelose = [e for e in cmds if not e["cmd"].startswith("CHOOSE ")]
    if forcelose and cmds[-len(forcelose):] != forcelose:
        raise ReconError(f"{side}: a FORCELOSE is not the side's last command")
    st = {"req_n": 0, "req_err": False, "ptr": 0, "last": None}

    def answer(st):
        if st["ptr"] < len(mine):
            e = mine[st["ptr"]]
            e["upd"], st["last"] = st["req_n"], e
            st["ptr"] += 1
        else:
            st["last"] = None

    for slot, text in chunks:
        if slot != side:
            continue
        for ln in text.split("\n"):
            if ln.startswith("|request|") and len(ln) > len("|request|"):
                if st["req_err"]:
                    st["req_err"] = False
                else:
                    st["req_n"] += 1
                try:
                    wait = bool(json.loads(ln[len("|request|"):]).get("wait"))
                except ValueError:
                    wait = False
                if wait:
                    st["last"] = None
                else:
                    answer(st)
            if ln.startswith(INVALID):
                if st["last"] is None:
                    raise ReconError(f"{side}: an [Invalid choice] with no command answered")
                st["last"]["dropped"] = True
                answer(st)
            if ln.startswith(UNAVAILABLE):
                st["req_err"] = True
    if st["ptr"] != len(mine):
        raise ReconError(f"{side}: {len(mine) - st['ptr']} of {len(mine)} commands answer no request")
    for e in forcelose:
        e["upd"] = st["req_n"]


def reconstruct(cap):
    """The battle's choice log from its capture: ``{start, teams, entries, ours}``; ``entries`` = one dict per
    ``CHOOSE`` / ``FORCELOSE`` (``w`` its write index, ``side``, ``upd`` the writing side's update, ``cmd``,
    ``dropped``: an ``[Invalid choice]`` deleted it), ``ours[side]`` = that side's ``CHOOSE`` entries in write order
    (its decisions, a dropped one included)."""
    cmds = cap["commands"]
    if not cmds or not cmds[0].startswith("START "):
        raise ReconError("the capture's first command is not START")
    start = json.loads(cmds[0][len("START "):])
    teams = {"p1": str(start["p1"]["team"]), "p2": str(start["p2"]["team"])}
    by_side = {"p1": [], "p2": []}
    entries = []
    for w, c in enumerate(cmds):
        if c.startswith(("CHOOSE ", "FORCELOSE ")):
            side = c.split(" ", 2)[1]
            if side not in by_side:
                raise ReconError(f"command {w} {c!r}: no side")
            e = {"w": w, "side": side, "cmd": c, "upd": None, "dropped": False}
            entries.append(e)
            by_side[side].append(e)
    for side in ("p1", "p2"):
        _tag_side(side, by_side[side], cap["chunks"])
    return {"start": start, "teams": teams, "entries": entries,
            "ours": {s: [e for e in by_side[s] if e["cmd"].startswith("CHOOSE ")] for s in ("p1", "p2")}}


def cut(rc, side, i):
    """``(log, at, update)``: ``search_side.honest_log`` at ``side``'s decision ``i`` (its i-th written ``CHOOSE``)."""
    from main.anchors.search_side import _seed_text, core_command

    me = rc["ours"][side][i]
    u, w = me["upd"], me["w"]
    rows = [e for e in rc["entries"] if not e["dropped"] and e["w"] < w
            and (e["upd"] < u or (e["upd"] == u and e["side"] == side))]
    rows.sort(key=lambda e: (e["upd"], 0 if e["side"] == "p1" else 1, e["w"]))
    cmds = [core_command(e["cmd"], rc["teams"]) for e in rows]
    if any(c.startswith("FORCELOSE") for c in cmds):
        raise ReconError(f"decision {i}: a forfeit is in the log before it")
    st = rc["start"]
    log = {"format_id": str(st["formatid"]), "seed": _seed_text(st.get("seed")),
           "names": [str(st["p1"]["name"]), str(st["p2"]["name"])],
           "teams": [str(st["p1"]["team"]), str(st["p2"]["team"])], "cmds": cmds}
    return log, len(cmds), u


def _loads(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def cell_inputs(cell_dir):
    """(capture, banked decision records) of one cell, never another file of it."""
    caps = sorted(Path(cell_dir, "capture").glob("battle-*.json"))
    if len(caps) != 1:
        raise ReconError(f"{len(caps)} captures")
    return json.loads(caps[0].read_text()), _loads(Path(cell_dir, "search_decisions.jsonl"))


def is_path(r):
    """A banked record the search was asked to make (searched or excluded): it carries a key."""
    return "key" in r


def battle_plan(cap, recs):
    """The battle's decisions joined to the banked records, every identity check done but the frame's (the
    searcher's root guard): ``{side, deal, decisions: [{rec, path, log?, at?, update?}], last: index of the last path
    decision}``."""
    from main.anchors.forced_branch import log_sha256
    from main.anchors.search_side import KEY_SCHEMA, deal_key

    sides = {r["side"] for r in recs}
    if len(sides) != 1:
        raise ReconError(f"the banked records name sides {sorted(sides)}")
    side = sides.pop()
    rc = reconstruct(cap)
    ours = rc["ours"][side]
    if len(ours) != len(recs):
        raise ReconError(f"{len(ours)} decisions of {side} in the capture, {len(recs)} banked records")
    deal = deal_key(rc["start"], side)
    plan = []
    for i, r in enumerate(recs):
        if int(r["n"]) != i and not any(e["dropped"] for e in ours):
            raise ReconError(f"banked record {i} has n {r['n']}")
        x = {"rec": r, "path": is_path(r)}
        if x["path"]:
            log, at, upd = cut(rc, side, i)
            want = f"{KEY_SCHEMA}|{deal}|t{int(r['turn'])}|n{int(r['n'])}"
            for what, got, bank in (("log_sha256", log_sha256(log), r.get("log_sha256")), ("at", at, r.get("at")),
                                    ("update", upd, r.get("update")), ("key", want, r.get("key")),
                                    ("deal", deal, r.get("deal"))):
                if got != bank:
                    raise ReconError(f"decision {i} (t{r['turn']} n{r['n']}): {what} {got!r}, banked {bank!r}")
            if r.get("rescore_sample") and r.get("log") != log:
                raise ReconError(f"decision {i}: the rebuilt log is not the banked re-score sample's")
            x.update(log=log, at=at, update=upd)
        plan.append(x)
    path = [j for j, x in enumerate(plan) if x["path"]]
    return {"side": side, "deal": deal, "decisions": plan, "last": path[-1] if path else None, "rc": rc}


def masks(plan, from_turn, engine):
    """Every search-path decision's (mask, tokens), off ``honest_inv.sequence`` on the last one; the sequence's
    (turn, n) set must be the banked path set."""
    from main.value_search import honest_inv as INV

    if plan["last"] is None:
        return {}
    x = plan["decisions"][plan["last"]]
    r = x["rec"]
    dec = INV.Decision(log=x["log"], at=x["at"], side=plan["side"], turn=int(r["turn"]), n=int(r["n"]), mask=[],
                       tokens={}, key=r["key"])
    seq = INV.sequence(dec, from_turn, engine)
    got = {(d.turn, d.n): d for d in seq}
    want = {(int(y["rec"]["turn"]), int(y["rec"]["n"])) for y in plan["decisions"] if y["path"]}
    if set(got) != want:
        raise ReconError(f"the sequence's path decisions {sorted(set(got) ^ want)[:6]} differ from the banked ones")
    return {k: (d.mask, d.tokens) for k, d in got.items()}


# ------------------------------------------------------------------------------------------------ the sample
def complete_cells(cells):
    """Per panel, the cells with exactly one capture and a non-empty decision log (never a ``.partial.`` dir)."""
    out = {p: [] for p in PANELS}
    for name in sorted(os.listdir(cells)):
        m = PAT.match(name)
        if not m:
            continue
        d = Path(cells, name)
        dl = d / "search_decisions.jsonl"
        if len(list((d / "capture").glob("battle-*.json"))) == 1 and dl.is_file() and dl.stat().st_size > 0:
            out[m.group(1)].append(name)
    return out


def rank(salt, name):
    return hashlib.sha256(f"{salt}:{name}".encode()).hexdigest()


def sample(a):
    if not a.salt:
        sys.exit("--salt is required (the registration names it)")
    cells = complete_cells(a.cells)
    pick = {p: sorted(v, key=lambda n: rank(a.salt, n))[:a.per_panel] for p, v in cells.items()}
    for p, v in pick.items():
        if len(v) < a.per_panel:
            sys.exit(f"{p}: {len(v)} complete cells, {a.per_panel} asked")
    order = [pick[p][i] for i in range(a.per_panel) for p in PANELS]      # interleaved: a partial run stays balanced
    doc = {"schema": "q19_stage0_sample_v1", "salt": a.salt, "per_panel": a.per_panel,
           "cells_dir": str(Path(a.cells).resolve()), "complete": {p: len(v) for p, v in cells.items()},
           "rank": "sha256(f'{salt}:{cell_name}') ascending, the first per_panel of each panel", "panels": pick,
           "order": order}
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"complete": doc["complete"], "battles": len(order)}))


# ---------------------------------------------------------------------------------------------------- the run
def validator_cpu():
    """The CPU-s of the process's shared Showdown validator child so far (stage 1's FINDING S'H-9)."""
    from main.value_search import world_legality as WL

    v = WL._SHARED.get(WL.FORMAT)
    p = getattr(v, "_proc", None)
    if p is None or p.poll() is not None:
        return 0.0
    try:
        f = Path(f"/proc/{p.pid}/stat").read_text().rsplit(")", 1)[1].split()
    except OSError:
        return 0.0
    return (int(f[11]) + int(f[12])) / os.sysconf("SC_CLK_TCK")


def _rows(d):
    p = Path(d) / "rows.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def run(a):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import torch

    torch.set_num_threads(1)
    from agents.battle.core_obs import wrap_row
    from main.anchors.core_side import CorePolicy, load_policy
    from main.anchors.search_presets import canonical, parse
    from main.anchors.search_side import LiveSearcher
    from main.anchors.server import engine_stamp
    from main.anchors.session import OurSideState
    from main.value_search import honest_worlds as HW
    from main.value_search import live_identity as LI
    from main.value_search.search import cpu_all
    from main.value_search.sign_gate import GateScorer

    preset = canonical(a.preset)
    _p, from_turn = parse(preset)
    if a.salt:
        HW.KEY_TAG = f"sprime_honest_v1:replicate:{a.salt}"   # a REPLICATE: every world draw and fresh die moved
    if hashlib.sha256(Path(a.model).read_bytes()).hexdigest() != a.model_sha256:
        sys.exit("the model's bytes are not the registered ones")
    smp = json.loads(Path(a.sample).read_text())
    cells = Path(smp["cells_dir"])
    eng = engine_stamp()
    if a.engine and eng != a.engine:
        sys.exit(f"the engine stamp is {eng!r}, {a.engine!r} asked")
    model = load_policy(str(a.model), "cpu", "bare", OurSideState())
    policy = CorePolicy(model, stochastic=False, temperature=1.0)
    fields = LI.compare_fields(preset)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_f = out / "rows.jsonl"
    meta = {"preset": preset, "salt": a.salt, "sample": str(Path(a.sample).resolve()), "sample_salt": smp["salt"],
            "model_sha256": a.model_sha256, "engine": eng}
    mf = out / "meta.json"
    if mf.exists():
        old = json.loads(mf.read_text())
        if {k: old.get(k) for k in meta} != meta:
            sys.exit(f"{out} was run as {old}, now {meta}")
    else:
        mf.write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    done, done_n = set(), {}
    for r in _rows(out):
        if r.get("group_done"):
            done.add(r["group"])
        elif "j" in r:
            done_n.setdefault(r["group"], set()).add(int(r["j"]))
    carries = out / "carries"
    carries.mkdir(exist_ok=True)
    i, k = (int(x) for x in a.shard.split("/"))
    todo = [g for j, g in enumerate(smp["order"]) if j % k == i and g not in done]
    if a.max_battles:
        todo = todo[:a.max_battles]

    def emit(row):
        with open(rows_f, "a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    for name in todo:
        panel = PANELS[PAT.match(name).group(1)]
        s = LiveSearcher(model, policy, preset, Path("/dev/null"))
        guard = GateScorer(a.model, searcher=s)
        g0 = time.monotonic()
        try:
            cap, recs = cell_inputs(cells / name)
            plan = battle_plan(cap, recs)
            mk = masks(plan, from_turn, s.engine)
        except (ReconError, ValueError, KeyError) as exc:      # a battle failing a hard check: a named row
            s.close()
            emit({"group": name, "panel": panel, "group_done": True, "group_error": f"{type(exc).__name__}: "
                  f"{str(exc)[:400]}"})
            continue
        cfile = carries / (hashlib.sha256(name.encode()).hexdigest()[:24] + ".json.gz")
        have = done_n.get(name, set())
        if have and cfile.exists() and hasattr(s.sampler, "seed"):
            s.sampler.seed(gzip.decompress(cfile.read_bytes()).decode())
        err = None
        for j, x in enumerate(plan["decisions"]):
            r = x["rec"]
            turn, n = int(r["turn"]), int(r["n"])
            if j in have:
                continue
            if not x["path"]:
                emit({"group": name, "panel": panel, "j": j, "turn": turn, "n": n, "path": False,
                      "banked": {kk: r.get(kk) for kk in BANKED}})
                continue
            mask, tokens = mk[(turn, n)]
            rec = {"log": x["log"], "at": x["at"], "side": plan["side"], "turn": turn, "n": n, "mask": mask,
                   "tokens": tokens, "key": r["key"], "frame_sha256": r["frame_sha256"],
                   "pi_argmax": int(r["pi_argmax"])}
            v0, c0, w0 = validator_cpu(), cpu_all(), time.monotonic()
            try:
                _truth, d = guard.root(rec)
                res = s._search(rec["log"], rec["at"], rec["side"], np.array(wrap_row(d["frame"]), dtype=np.float32),
                                np.asarray(mask, dtype=np.int8), d, rec["key"], rec["pi_argmax"])
            except Exception as exc:  # noqa: BLE001 -- a refused root or forward: a named group error
                err = f"decision t{turn} n{n}: {type(exc).__name__}: {str(exc)[:400]}"
                break
            row = {"group": name, "panel": panel, "j": j, "turn": turn, "n": n, "path": True, "key": r["key"],
                   "at": x["at"], "update": x["update"], "n_legal": int(r["n_legal"]), "pi_argmax": int(r["pi_argmax"]),
                   "engine": eng, "preset": preset, "salt": a.salt, "res": {kk: res.get(kk) for kk in FIELDS},
                   "banked": {kk: r.get(kk) for kk in BANKED}}
            h = res.get("honest") or {}
            row.update(inc=h.get("inc"), chain=h.get("chain"), fails=h.get("fails"),
                       kinds=[w.get("kind") for w in h.get("worlds") or []])
            row["cpu_all_s"] = round(cpu_all() - c0, 3)
            row["validator_cpu_s"] = round(validator_cpu() - v0, 3)
            row["wall_s"] = round(time.monotonic() - w0, 3)
            row["rebuilt"] = getattr(s.sampler, "last_rebuilt", None)
            row["phases"] = dict(getattr(s.sampler, "last_phases", None) or {})
            # the banked record against this one, field by field (meaningful at REF unsalted: the identity)
            diff = {dd["field"] for dd in LI.compare(r, {"search": res}, fields)}
            row["identity"] = {f: f not in diff for f in fields}
            emit(row)
            store = getattr(s.sampler, "store", None)
            if store:                     # the carry the decision returned, atomically (a resume seeds it)
                tmp = cfile.with_suffix(".tmp")
                tmp.write_bytes(gzip.compress(store[next(reversed(store))].encode(), mtime=0))
                os.replace(tmp, cfile)
            print(json.dumps({"g": name, "t": turn, "n": n, "s": res.get("searched"), "cpu": row["cpu_all_s"],
                              "wall": row["wall_s"], "id": all(row["identity"].values())}), flush=True)
        s.close()
        if err is not None:
            emit({"group": name, "panel": panel, "group_done": True, "group_error": err})
            continue
        emit({"group": name, "panel": panel, "group_done": True, "decisions": len(plan["decisions"]),
              "path": sum(1 for x in plan["decisions"] if x["path"]), "wall_s": round(time.monotonic() - g0, 3)})


# ---------------------------------------------------------------------------------------------- the summary
def F(x):
    """An exact Fraction of a JSON number (its decimal spelling: the rows' CPU-s are rounded to ms)."""
    return Fraction(str(x))


def judge(value, op, bar):
    """``{value, bar, op, verdict}``: PASS / FAIL, ``near`` (within :data:`NEAR` of the bar) counting as not passing."""
    if abs(value - bar) <= NEAR:
        v = "near"
    else:
        v = "PASS" if (value <= bar if op == "<=" else value >= bar) else "FAIL"
    return {"value": float(value), "value_exact": str(value), "op": op, "bar": float(bar), "bar_exact": str(bar),
            "verdict": v, "pass": v == "PASS"}


def arm_of(d):
    rows = _rows(d)
    done = {r["group"] for r in rows if r.get("group_done") and not r.get("group_error")}
    errs = {r["group"]: r["group_error"] for r in rows if r.get("group_error")}
    by = {}
    for r in rows:
        if "group_done" not in r and r["group"] in done:
            by.setdefault(r["group"], []).append(r)
    for v in by.values():
        v.sort(key=lambda r: int(r["j"]))
    meta = json.loads((Path(d) / "meta.json").read_text())
    return {"dir": str(d), "meta": meta, "battles": by, "errors": errs}


def cpu_of(r, meter):
    """A path row's search CPU on the named meter, plus the validator child's."""
    base = r["cpu_all_s"] if meter == "cpu_all" else r["res"]["cpu_s"]
    return F(base) + F(r["validator_cpu_s"])


def measure(arm, battles, meter, ref_banked_walls=False):
    """Counts and exact sums of one arm on ``battles``: per panel and pooled U."""
    from main.reads import timer_model as TM

    acc = {}
    games = []
    for g in battles:
        rows = arm["battles"][g]
        P = rows[0]["panel"]
        clock = []
        for r in rows:
            if r["path"]:
                searched = bool(r["res"]["searched"])
                excluded = r["res"]["why"] == "excluded"
                wall = r["banked"]["wall_s"] if ref_banked_walls else r["wall_s"]
            else:
                searched, excluded, wall = False, False, 0.0
            clock.append({"turn": int(r["turn"]), "searched": searched, "excluded": excluded, "wall": float(wall or 0)})
        games.append((P, clock))
        for key in (P, "U"):
            a = acc.setdefault(key, {"games": 0, "path": 0, "searched": 0, "excluded": 0, "cpu_searched": Fraction(0),
                                     "cpu_excluded": Fraction(0), "validator": Fraction(0), "walls": [],
                                     "agree": 0, "both": 0})
            a["games"] += 1
            for r in rows:
                if not r["path"]:
                    continue
                a["path"] += 1
                c = cpu_of(r, meter)
                a["validator"] += F(r["validator_cpu_s"])
                a["walls"].append(r["wall_s"])
                if r["res"]["searched"]:
                    a["searched"] += 1
                    a["cpu_searched"] += c
                elif r["res"]["why"] == "excluded":
                    a["excluded"] += 1
                    a["cpu_excluded"] += c
                if r["res"]["searched"] and r["banked"]["searched"]:
                    a["both"] += 1
                    a["agree"] += int(int(r["res"]["played"]) == int(r["banked"]["played"]))
    timer = {pol: TM.tally(games, pol) for pol in (TM.NET_POLICY, "bank_r10")}
    return acc, timer, TM.replay(games)


def _q(xs, f):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(f * len(xs)))] if xs else None


def view(acc, timer):
    out = {}
    for P, a in acc.items():
        g = a["games"]
        cpu = a["cpu_searched"] + a["cpu_excluded"]
        out[P] = {"games": g, "path": a["path"], "searched": a["searched"], "excluded": a["excluded"],
                  "excl_share": round(a["excluded"] / a["path"], 4) if a["path"] else None,
                  "search_cpu_s_per_game": round(float(cpu / g), 2),
                  "searched_cpu_s_per_game": round(float(a["cpu_searched"] / g), 2),
                  "excluded_cpu_s_per_game": round(float(a["cpu_excluded"] / g), 2),
                  "validator_cpu_s_per_game": round(float(a["validator"] / g), 2),
                  "cpu_s_per_path_decision": round(float(cpu / a["path"]), 3) if a["path"] else None,
                  "wall_s": {q: _q(a["walls"], f) for q, f in (("p50", .5), ("p90", .9), ("p99", .99),
                                                               ("p999", .999), ("max", 1.0))},
                  "agree_banked_ref": [a["agree"], a["both"]],
                  "timer": {pol: t.get(P) for pol, t in timer.items()}}
    return out


def rule(arm_acc, arm_timer, ref_acc, floor_acc):
    """THE SELECTION RULE (DRAFT §3), pooled U, every comparison exact."""
    from main.reads import timer_model as TM

    U, R = arm_acc["U"], ref_acc["U"]
    t = arm_timer[TM.NET_POLICY]["U"]
    crit = {}
    i_forfeit = {"value": t["forfeit"], "op": "==", "bar": 0, "verdict": "PASS" if t["forfeit"] == 0 else "FAIL"}
    i_forfeit["pass"] = i_forfeit["verdict"] == "PASS"
    i_cut = judge(Fraction(t["cut_searched"], t["searched"]) if t["searched"] else Fraction(1), "<=", Fraction(1, 50))
    crit["i"] = {"policy": TM.NET_POLICY, "forfeit_games": i_forfeit, "cut_share_of_searched": i_cut,
                 # the criterion's own reading, uniform with ii-iv: the cut share against 2 %, failed by any forfeit
                 "value": i_cut["value"], "value_exact": i_cut["value_exact"], "op": "<=", "bar": i_cut["bar"],
                 "verdict": i_cut["verdict"] if i_forfeit["pass"] else "FAIL (forfeit)",
                 "pass": i_forfeit["pass"] and i_cut["pass"],
                 "reported_bank_r10": arm_timer["bank_r10"]["U"]}
    arm_cpu = (U["cpu_searched"] + U["cpu_excluded"]) / U["games"]
    ref_cpu = (R["cpu_searched"] + R["cpu_excluded"]) / R["games"]
    crit["ii"] = judge(arm_cpu / ref_cpu, "<=", Fraction(7, 10))
    crit["ii"].update(arm_cpu_s_per_game=float(arm_cpu), ref_cpu_s_per_game=float(ref_cpu))
    if floor_acc is None or not floor_acc["U"]["both"] or not U["both"]:
        crit["iii"] = {"verdict": "no floor" if floor_acc is None else "no decisions both searched", "pass": False}
    else:
        fl = Fraction(floor_acc["U"]["agree"], floor_acc["U"]["both"])
        crit["iii"] = judge(Fraction(U["agree"], U["both"]), ">=", fl - Fraction(1, 20))
        crit["iii"].update(agree=[U["agree"], U["both"]], floor=[floor_acc["U"]["agree"], floor_acc["U"]["both"]])
    crit["iv"] = judge(Fraction(U["excluded"], U["path"]), "<=", Fraction(R["excluded"], R["path"]) + Fraction(1, 100))
    crit["iv"].update(arm=[U["excluded"], U["path"]], ref=[R["excluded"], R["path"]])
    return {"criteria": crit, "pass": all(crit[c]["pass"] for c in ("i", "ii", "iii", "iv")), "cpu": arm_cpu}


#: the tie order of the draft's rule: the simpler lever first (K4, then CAP), then any other
TIE_ORDER = ("r0s4_honest_inc_k4", "r0s4_honest_inc_cap")


def summarize(a):
    arms = [arm_of(d) for d in a.arm]
    ref = arm_of(a.ref)
    floor = arm_of(a.floor) if a.floor else None
    every = [ref] + arms + ([floor] if floor else [])
    common = set(ref["battles"])
    for x in every:
        common &= set(x["battles"])
    battles = sorted(common)
    out = {"schema": "q19_stage0_summary_v1", "cpu_meter": a.cpu_meter, "battles": len(battles),
           "battles_by_panel": {P: sum(1 for g in battles if ref["battles"][g][0]["panel"] == P)
                                for P in ("H", "F", "P72")},
           "dirs": {x["dir"]: {"meta": x["meta"], "completed": len(x["battles"]), "group_errors": x["errors"]}
                    for x in every},
           "arms": {}}
    if not battles:
        out["verdict"] = "no battle completed by every dir"
        Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
        print(json.dumps(out, indent=1, sort_keys=True))
        return
    ref_acc, ref_timer, ref_rep = measure(ref, battles, a.cpu_meter)
    _bacc, banked_timer, banked_rep = measure(ref, battles, a.cpu_meter, ref_banked_walls=True)
    floor_acc = None
    out["ref"] = {"preset": ref["meta"]["preset"], **view(ref_acc, ref_timer)}
    out["ref_replay"] = ref_rep
    out["ref_banked_walls_replay"] = {"note": "REF's re-run decisions with S'-HONEST's banked walls (K = 14 box): "
                                              "the wall regime's calibration", "timer": banked_timer,
                                      "replay": banked_rep}
    out["identity_ref"] = identity(ref, battles)
    if floor:
        floor_acc, floor_timer, _r = measure(floor, battles, a.cpu_meter)
        out["floor"] = {"preset": floor["meta"]["preset"], "salt": floor["meta"]["salt"],
                        **view(floor_acc, floor_timer)}
    passing = []
    for x in arms:
        acc, timer, rep = measure(x, battles, a.cpu_meter)
        r = rule(acc, timer, ref_acc, floor_acc)
        name = x["meta"]["preset"]
        out["arms"][x["dir"]] = {"preset": name, "salt": x["meta"]["salt"], **view(acc, timer), "replay": rep,
                                 "rule": {k: v for k, v in r.items() if k != "cpu"}}
        if r["pass"]:
            base = name.split("@")[0]
            passing.append((r["cpu"], TIE_ORDER.index(base) if base in TIE_ORDER else len(TIE_ORDER), name, x["dir"]))
    passing.sort()
    out["selection"] = ({"pick": passing[0][2], "dir": passing[0][3], "passing": [p[2] for p in passing]}
                        if passing else {"pick": None, "finding": "no arm passes: the read ends at stage 0"})
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps(out["selection"], indent=1))


def identity(arm, battles):
    """Per compared field, how many path decisions of ``arm`` equal the banked record (REF unsalted: the identity)."""
    out = {}
    n = 0
    for g in battles:
        for r in arm["battles"][g]:
            if not r["path"]:
                continue
            n += 1
            for f, ok in r["identity"].items():
                out.setdefault(f, 0)
                out[f] += int(ok)
    return {"decisions": n, "equal": out}


# ---------------------------------------------------------------------------------------------- self-checks
def _syn_team(names):
    return "]".join(f"{nm}|{sp}|Leftovers|Ability|tackle,growl,,|Hardy|84,84,84,84,84,84|||||" for nm, sp in names)


def _syn(commands, chunks):
    st = {"formatid": "gen3ou", "seed": [1, 2, 3, 4],
          "p1": {"name": "A", "team": _syn_team([("Alpha", "Charizard"), ("Beta", "Blissey")])},
          "p2": {"name": "B", "team": _syn_team([("Gamma", "Skarmory"), ("Delta", "Tyranitar")])}}
    return {"tag": "t", "seed": st["seed"], "commands": ["START " + json.dumps(st)] + commands + ["END"],
            "chunks": chunks}


REQ = '|request|{"active":[{}],"rqid":1}'
WAIT = '|request|{"wait":true,"rqid":1}'


def selfcheck(_a=None):
    # 1. p2 writes before p1 in one update: the cut orders (update, p1 before p2); p1's decision at update 2 never
    #    holds p2's update-2 choice; p2's decision at update 2 never holds p1's update-2 choice
    cap = _syn(["CHOOSE p2 move tackle", "CHOOSE p1 move growl", "CHOOSE p2 move growl", "CHOOSE p1 move tackle"],
               [["p1", REQ], ["p2", REQ], ["p1", "|turn|2\n" + REQ], ["p2", "|turn|2\n" + REQ]])
    rc = reconstruct(cap)
    assert [e["upd"] for e in rc["entries"]] == [1, 1, 2, 2], rc["entries"]
    log, at, u = cut(rc, "p1", 1)
    assert (log["cmds"], at, u) == (["CHOOSE p1 move growl", "CHOOSE p2 move tackle"], 2, 2), log
    log, at, u = cut(rc, "p2", 1)
    assert log["cmds"] == ["CHOOSE p1 move growl", "CHOOSE p2 move tackle"] and u == 2, log
    log, _at, _u = cut(rc, "p2", 0)
    assert log["cmds"] == [], log
    # 2. an [Invalid choice] dropped: p1's bad command leaves the log; the re-choice answers the SAME update
    cap = _syn(["CHOOSE p1 move flamethrower", "CHOOSE p1 move tackle", "CHOOSE p2 move tackle",
                "CHOOSE p1 move growl", "CHOOSE p2 move growl"],
               [["p1", REQ], ["p2", REQ], ["p1", "|error|[Invalid choice] Can't move: no such move"],
                ["p1", "|turn|2\n" + REQ], ["p2", "|turn|2\n" + REQ]])
    rc = reconstruct(cap)
    assert [(e["upd"], e["dropped"]) for e in rc["entries"]] == [(1, True), (1, False), (1, False), (2, False),
                                                                 (2, False)], rc["entries"]
    assert len(rc["ours"]["p1"]) == 3
    log, at, u = cut(rc, "p1", 1)            # the re-choice: the dropped one is not in its log
    assert log["cmds"] == [] and u == 1, log
    log, at, u = cut(rc, "p1", 2)
    assert log["cmds"] == ["CHOOSE p1 move tackle", "CHOOSE p2 move tackle"] and u == 2, log
    # 3. an [Unavailable choice] re-asked: the command STAYS, the re-request is the same update, and the re-choice's
    #    log holds our own earlier choice of our update (never the opponent's)
    cap = _syn(["CHOOSE p2 move tackle", "CHOOSE p1 switch beta", "CHOOSE p1 move tackle", "CHOOSE p1 move growl",
                "CHOOSE p2 move growl"],
               [["p1", REQ], ["p2", REQ], ["p1", "|error|[Unavailable choice] Can't switch: trapped\n" + REQ],
                ["p1", "|turn|2\n" + REQ], ["p2", "|turn|2\n" + REQ]])
    rc = reconstruct(cap)
    assert [e["upd"] for e in rc["entries"]] == [1, 1, 1, 2, 2], rc["entries"]
    log, at, u = cut(rc, "p1", 1)
    assert log["cmds"] == ["CHOOSE p1 switch Beta"] and (at, u) == (1, 1), log      # re-spelled by core_command
    log, at, u = cut(rc, "p1", 2)
    assert log["cmds"] == ["CHOOSE p1 switch Beta", "CHOOSE p1 move tackle", "CHOOSE p2 move tackle"], log
    # 4. a forced switch alone: p2 faints, p1 waits (its update still counts), p2's switch is update 2
    cap = _syn(["CHOOSE p1 move tackle", "CHOOSE p2 move tackle", "CHOOSE p2 switch tyranitar",
                "CHOOSE p1 move growl", "CHOOSE p2 move growl"],
               [["p1", REQ], ["p2", REQ], ["p1", "|faint|p2a: Gamma\n" + WAIT],
                ["p2", '|faint|p2a: Gamma\n|request|{"forceSwitch":[true],"rqid":3}'],
                ["p1", "|turn|2\n" + REQ], ["p2", "|turn|2\n" + REQ]])
    rc = reconstruct(cap)
    assert [e["upd"] for e in rc["entries"]] == [1, 1, 2, 3, 3], rc["entries"]
    log, at, u = cut(rc, "p1", 1)
    assert log["cmds"] == ["CHOOSE p1 move tackle", "CHOOSE p2 move tackle", "CHOOSE p2 switch Delta"] and u == 3
    log, at, u = cut(rc, "p2", 1)
    assert log["cmds"] == ["CHOOSE p1 move tackle", "CHOOSE p2 move tackle"] and u == 2, log
    # 5. a command answering no request is refused
    try:
        reconstruct(_syn(["CHOOSE p1 move tackle", "CHOOSE p1 move growl"], [["p1", REQ]]))
        raise AssertionError("an unanswered command was accepted")
    except ReconError:
        pass
    # 6. the selection rule's arithmetic: exact, rule 8's band
    assert judge(Fraction(1, 50), "<=", Fraction(1, 50))["verdict"] == "near"
    assert judge(Fraction(1, 50) + Fraction(4, 10000), "<=", Fraction(1, 50))["verdict"] == "near"
    assert judge(Fraction(1, 50) + Fraction(6, 10000), "<=", Fraction(1, 50))["verdict"] == "FAIL"
    assert judge(Fraction(1, 50) - Fraction(6, 10000), "<=", Fraction(1, 50))["verdict"] == "PASS"
    assert judge(Fraction(1, 50) - Fraction(5, 10000), "<=", Fraction(1, 50))["verdict"] == "near"
    assert judge(Fraction(58, 100), ">=", Fraction(63, 100) - Fraction(1, 20))["verdict"] == "near"
    assert judge(Fraction(5806, 10000), ">=", Fraction(63, 100) - Fraction(1, 20))["verdict"] == "PASS"
    assert judge(Fraction(5794, 10000), ">=", Fraction(63, 100) - Fraction(1, 20))["verdict"] == "FAIL"
    assert judge(Fraction(7, 10), "<=", Fraction(7, 10))["pass"] is False
    assert judge(Fraction(69, 100), "<=", Fraction(7, 10))["pass"] is True
    print("selfcheck: OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--cells", required=True)
    s.add_argument("--salt", required=True, help="the registered salt of the sample's hash (never defaulted)")
    s.add_argument("--per-panel", type=int, required=True)
    s.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--preset", required=True, help="a registered <preset>@<turn>, e.g. r0s4_honest_inc@10")
    r.add_argument("--model", required=True)
    r.add_argument("--model-sha256", required=True)
    r.add_argument("--sample", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--salt", default="", help="a re-draw replicate: replaces honest_worlds.KEY_TAG (REF's floor)")
    r.add_argument("--engine", default="", help="refuse unless the engine stamp is this one")
    r.add_argument("--shard", default="0/1")
    r.add_argument("--max-battles", type=int, default=0, help="at most this many battles this invocation (0: all)")
    m = sub.add_parser("summarize")
    m.add_argument("--arm", action="append", default=[], required=True)
    m.add_argument("--ref", required=True)
    m.add_argument("--floor", default="")
    m.add_argument("--cpu-meter", choices=("cpu_all", "cpu_s"), required=True,
                   help="the per-decision search CPU of criterion (ii): cpu_all (every thread and reaped child, "
                        "search.cpu_all) or cpu_s (the record's: its thread and children); + the validator's either way")
    m.add_argument("--out", required=True)
    sub.add_parser("selfcheck")
    a = ap.parse_args()
    {"sample": sample, "run": run, "summarize": summarize, "selfcheck": selfcheck}[a.cmd](a)
