"""REVIEW G (data quality and selection bias of the X59-S read): the CPU checks behind REVIEW_G_data_bias.md.
Deterministic; read-only on the archive, the corpus and the students.

    PYTHONPATH=$PWD/src CUDA_VISIBLE_DEVICES="" nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh --max-s 600 8 \
        $PY designs/research_state/measurements/x59s_matched_students_2026-10-25/review_g_checks.py [split|profile|logs|nll|floors|pairs]

* ``split``: r2's account split at the BATTLE level: VALIDATION rows whose battle also gives TRAIN rows (the other
  seat), and S-S's game split likewise.
* ``profile``: the rows' turn / forced-switch / ending / rows-a-battle profile, S-H (r2) against S-S (ss_rows_v2).
* ``logs``: over the 27,586 kept battles' logs: (a) the panel residual, every SEEN mon of a side (not only those with
  a revealed move, the rowset rule) fitted to the SCORED panels' teams (H, F, P72 and the retired heldout72;
  ``forum216b`` is not read); (b) the battles holding a volatile Showdown ends silently (``main.x50_r1.fr2_exclude``'s
  patterns), whose rows r2 encodes with the pre-``d8d674b99`` reader; (c) duplicate battle ids and duplicate
  (accounts, turns, ending) records among the kept battles (a one-seat battle keys by ONE account, so most of these
  groups are one player's unrelated games: ``pairs`` is the two-seat count).
* ``floors``: ``nll``'s P* floors alone (no forward), rewriting ``nll.human_val``.
* ``pairs``: repeated meetings of the same two kept accounts, and the duplicate groups of two-seat battles only.
* ``nll``: why the arms' own-label validation NLLs are not comparable: P*'s own entropy (the floor a perfect student of
  a T = 1 sample reaches) on each arm's validation rows, and the set-label (hidden) rows' share of each NLL.

Writes ``results/review_g_checks.json`` (each sub-command updates its own key).
"""
from __future__ import annotations

import collections
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

ARCH = Path.home() / "gen3ai_archive" / "x59s"
R2 = ARCH / "hr_spec_v1.r2" / "rows"
R1 = ARCH / "hr_spec_v1.r1"
SS = ARCH / "ss_rows_v2"
STUDENTS = ARCH / "read" / "students"
CORPUS = Path("/home/goodlad/dev/gen3ai/replays/showdown/gen3ou")
PSTAR = Path("/home/goodlad/dev/gen3ai/models/rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip")
SCORED = ("heldout72", "heldout72_hpfix", "forum72", "p72")
OUT = Path(__file__).resolve().parent / "results" / "review_g_checks.json"
SW = np.arange(11) <= 5                                   # actions 0-5 are switches (REVIEW E's rule)


def _manifest(d: Path) -> dict:
    return json.loads((d / "rows_manifest.json").read_text())


def _shards(d: Path):
    m = _manifest(d)
    tr = m["train_shards"]
    return (tr if isinstance(tr, list) else eval(tr)), m["val"]  # noqa: S307 (the manifest's own repr)


def _fields(d: Path, rec: dict, fields):
    z = np.load(d / f"{rec['name']}.npz")
    return {f: z[f] for f in fields}


def cmd_split() -> dict:
    out = {}
    for name, d in (("s_h_r2", R2), ("s_s", SS)):
        tr, va = _shards(d)
        tb = np.concatenate([_fields(d, r, ("bkey",))["bkey"] for r in tr])
        v = _fields(d, va, ("bkey", "acct", "seat"))
        shared = np.isin(v["bkey"], np.unique(tb))
        out[name] = {"val_rows": int(len(v["bkey"])), "val_battles": int(len(np.unique(v["bkey"]))),
                     "val_rows_in_a_battle_with_train_rows": int(shared.sum()),
                     "val_battles_with_train_rows": int(len(np.unique(v["bkey"][shared])))}
    return out


def cmd_profile() -> dict:
    out = {}
    for name, d in (("s_h_r2", R2), ("s_s", SS)):
        tr, _ = _shards(d)
        acc = collections.defaultdict(list)
        for r in tr:
            z = _fields(d, r, ("turn", "forced", "end", "bkey", "act", "hidden", "mask"))
            for k, v in z.items():
                acc[k].append(v)
        z = {k: np.concatenate(v) for k, v in acc.items()}
        lab = ~z["hidden"]
        t = z["turn"].astype(np.int64)
        nb = len(np.unique(z["bkey"]))
        out[name] = {"rows": int(len(t)), "battles": nb, "rows_per_battle": round(len(t) / nb, 2),
                     "turn_quantiles_10_25_50_75_90": [float(x) for x in np.percentile(t, [10, 25, 50, 75, 90])],
                     "share_turn_ge_30": round(float((t >= 30).mean()), 4),
                     "share_turn_ge_50": round(float((t >= 50).mean()), 4),
                     "forced_share": round(float(z["forced"].mean()), 4),
                     "switch_label_share_labelled": round(float(SW[z["act"][lab].astype(np.int64)].mean()), 4),
                     "legal_actions_mean": round(float(z["mask"].sum(1).mean()), 3),
                     "end_share": {k: round(float((z["end"] == c).mean()), 4)
                                   for k, c in (("ko", 0), ("forfeit", 1), ("timer", 2), ("tie", 3), ("none", 4))}}
    return out


def cmd_logs() -> dict:
    from agents.info_levers import revealed
    from main.human_data.rowset import key64
    from main.human_data.textpass import PanelMatcher, fits
    from main.x50_r1 import fr2_exclude

    pm = PanelMatcher(panels=SCORED)
    keep = [json.loads(x) for x in __import__("gzip").open(R1 / "keep.jsonl.gz", "rt")]
    if os.environ.get("REVIEW_G_LIMIT"):                     # a smoke of the first N battles only
        keep = keep[:int(os.environ["REVIEW_G_LIMIT"])]
    # rows of r2 TRAIN by (battle, seat): turn arrays for the stale-cell bound and the leakage row shares
    tr, va = _shards(R2)
    acc = collections.defaultdict(list)
    for r in list(tr) + [va]:
        z = _fields(R2, r, ("bkey", "seat", "turn"))
        for k, v in z.items():
            acc[k].append(v)
    rb, rs, rt = (np.concatenate(acc[k]) for k in ("bkey", "seat", "turn"))
    rows_by = collections.defaultdict(list)
    order = np.lexsort((rs, rb))
    rb, rs, rt = rb[order], rs[order], rt[order]
    cut = np.flatnonzero(np.diff(rb) | np.diff(rs)) + 1
    for s, e in zip(np.r_[0, cut], np.r_[cut, len(rb)]):
        rows_by[(int(rb[s]), int(rs[s]))] = rt[s:e]
    n_rows = len(rb)

    turn_re = re.compile(r"^\|turn\|(\d+)")
    vol_battles, vol_kinds, stale_rows_ub, vol_rows = 0, collections.Counter(), 0, 0
    fit_hist = {"own_actor": collections.Counter(), "opponent_of_actor": collections.Counter()}
    leak_rows = {"own_ge4": 0, "own_ge5": 0, "own_eq6": 0, "opp_ge4": 0, "opp_ge5": 0}
    species_hist = collections.Counter()
    ids = collections.Counter()
    recs = collections.Counter()
    missing = 0
    t0 = time.time()
    for k in keep:
        p = CORPUS / k["rel"]
        try:
            lines = p.read_text(errors="replace").splitlines()
        except OSError:
            missing += 1
            continue
        bk = key64(k["rel"])
        ids[re.sub(r"^.*-(\d+)\.log$", r"\1", k["rel"])] += 1
        recs[(tuple(sorted(k["acct"].values())), k["turns"], k["end"])] += 1
        # (b) the silently ending volatiles: first turn of a start line
        first, turn = None, 0
        for ln in lines:
            m = turn_re.match(ln)
            if m:
                turn = int(m.group(1))
                continue
            kind = fr2_exclude.match(ln)
            if kind is not None:
                vol_kinds[kind] += 1
                if first is None:
                    first = turn
        seats = {"p1": 0, "p2": 1}
        if first is not None:
            vol_battles += 1
            for s in k["seats"]:
                t = rows_by.get((bk, seats[s]))
                if t is not None:
                    vol_rows += len(t)
                    stale_rows_ub += int((t >= first).sum())
        # (a) every seen mon of each side, fitted to one scored-panel team
        side_fit = {}
        for s, viewer in (("p1", 1), ("p2", 0)):
            rv = revealed.from_lines(lines, viewer)
            count = collections.Counter()
            sp = collections.Counter()
            for sid, mr in rv.mons.items():
                c = pm.canon(revealed.toid(sid))
                mv = sorted(revealed.toid(x) for x in mr.moves)
                for (panel, i, ms) in pm.index.get(c, ()):
                    sp[(panel, i)] += 1
                    if fits(mv, mr.item, mr.ability, ms):
                        count[(panel, i)] += 1
            side_fit[s] = (max(count.values()) if count else 0, max(sp.values()) if sp else 0)
        for s in k["seats"]:
            o = "p2" if s == "p1" else "p1"
            fo, fs = side_fit[s][0], side_fit[o][0]
            fit_hist["own_actor"][fo] += 1
            fit_hist["opponent_of_actor"][fs] += 1
            species_hist[side_fit[s][1]] += 1
            t = rows_by.get((bk, seats[s]))
            n = 0 if t is None else len(t)
            leak_rows["own_ge4"] += n * (fo >= 4)
            leak_rows["own_ge5"] += n * (fo >= 5)
            leak_rows["own_eq6"] += n * (fo >= 6)
            leak_rows["opp_ge4"] += n * (fs >= 4)
            leak_rows["opp_ge5"] += n * (fs >= 5)
    seats_n = sum(fit_hist["own_actor"].values())
    return {
        "battles": len(keep), "missing_logs": missing, "seconds": round(time.time() - t0, 1),
        "r2_rows_matched": int(sum(len(v) for v in rows_by.values())), "r2_rows": int(n_rows),
        "panel_residual": {
            "rule": "every SEEN mon (no revealed move needed) of a side; fits = textpass.fits on what the log revealed; "
                    "max over one team of H, F, P72, heldout72",
            "actor_side_fit_hist": dict(sorted(fit_hist["own_actor"].items())),
            "opponent_side_fit_hist": dict(sorted(fit_hist["opponent_of_actor"].items())),
            "actor_side_species_only_hist": dict(sorted(species_hist.items())),
            "kept_seats": seats_n,
            "r2_row_share": {k: round(v / n_rows, 4) for k, v in leak_rows.items()},
        },
        "silent_volatiles": {
            "battles_flagged": vol_battles, "battle_share": round(vol_battles / len(keep), 4),
            "kinds": dict(vol_kinds), "r2_rows_in_flagged_battles": vol_rows,
            "r2_rows_at_or_after_first_start_UB": stale_rows_ub,
            "r2_row_share_UB": round(stale_rows_ub / n_rows, 4),
        },
        "duplicates": {"battle_ids_seen_twice": sum(1 for v in ids.values() if v > 1),
                       "same_accounts_turns_ending_twice": sum(1 for v in recs.values() if v > 1)},
    }


def cmd_pairs() -> dict:
    """Repeated meetings of the same two kept accounts (win-trading / alt farming would show here), and the
    (accounts, turns, ending) duplicate groups restricted to battles with BOTH seats kept (``logs``' count keys a
    one-seat battle by one account, so most of its groups are one player's unrelated games)."""
    import gzip

    keep = [json.loads(x) for x in gzip.open(R1 / "keep.jsonl.gz", "rt")]
    both = [k for k in keep if len(k["acct"]) == 2]
    pc = collections.Counter(tuple(sorted(k["acct"].values())) for k in both)
    rc = collections.Counter((tuple(sorted(k["acct"].values())), k["turns"], k["end"]) for k in both)
    return {"battles": len(keep), "both_seats_kept": len(both),
            "pairs_meeting": {str(t): {"pairs": sum(1 for c in pc.values() if c >= t),
                                       "battles": sum(c for c in pc.values() if c >= t)} for t in (2, 5, 10)},
            "max_meetings": max(pc.values()),
            "two_seat_same_accounts_turns_ending_groups": sum(1 for c in rc.values() if c > 1)}


def _softmax(lg, m):
    x = np.where(m, lg.astype(np.float64), -np.inf)
    x = x - x.max(1, keepdims=True)
    e = np.where(m, np.exp(x), 0.0)
    return e / e.sum(1, keepdims=True)


def _set_nll(lg, legal, target):
    p = _softmax(lg, legal.astype(bool))
    return -np.log(np.maximum((p * target).sum(1), 1e-300))


def _human_floors(with_rows: bool = False):
    """The human validation rows, P*'s stored logits on them, and the floors a perfect student of each P* teacher
    reaches there: P*'s entropy on a labelled row (unrestricted for S-P0, restricted to the consistent set for S-P),
    and on a hidden row (both arms train it on the SAME set label) P*'s own set-NLL of the legal moves."""
    _, hv = _shards(R2)
    H = _fields(R2, hv, ("rows",) * with_rows + ("mask", "act", "hidden", "cons", "moveset", "rid"))
    lz = np.load(ARCH / "hr_spec_v1.r2" / "labels_pstar" / "val.npz")
    assert np.array_equal(lz["rid"], H["rid"])
    hid = H["hidden"].astype(bool)
    legal = H["mask"].astype(bool)
    cons = H["cons"].astype(bool) & legal
    pl = _softmax(lz["logits"], legal)
    with np.errstate(invalid="ignore"):                    # a hidden row's consistent set may hold no action
        pr = _softmax(lz["logits"], cons)
    ent = lambda p: -(np.where(p > 0, p * np.log(np.maximum(p, 1e-300)), 0.0)).sum(1)  # noqa: E731
    hid_floor = -np.log(np.maximum((pl * H["moveset"]).sum(1), 1e-300))
    e0, er = ent(pl)[~hid], ent(np.nan_to_num(pr))[~hid]
    floors_h = {"pstar0_teacher_entropy_labelled": float(e0.mean()),
                "pstar_restricted_teacher_entropy_labelled": float(er.mean()),
                "hidden_rows_pstar_setnll": float(hid_floor[hid].mean()),
                "hidden_share": float(hid.mean()),
                "pstar0_floor_all_rows": float((e0.sum() + hid_floor[hid].sum()) / len(hid)),
                "pstar_floor_all_rows": float((er.sum() + hid_floor[hid].sum()) / len(hid)),
                "human_switch_share_labelled": float(SW[H["act"][~hid].astype(np.int64)].mean())}
    return H, hid, legal, cons, floors_h


def cmd_floors() -> dict:
    """``nll``'s P*-floor part alone (no forward): ``main`` writes it to ``nll.human_val``."""
    return _human_floors()[-1]


def cmd_nll() -> dict:
    from main.policy_spectrum.qhat import logits_and_values
    from main.policy_spectrum.reader import inference_globals, load_checkpoint

    _, sv = _shards(SS)
    S = _fields(SS, sv, ("rows", "mask", "act"))
    H, hid, legal, cons, floors_h = _human_floors(with_rows=True)
    ent = lambda p: -(np.where(p > 0, p * np.log(np.maximum(p, 1e-300)), 0.0)).sum(1)  # noqa: E731
    out = {"human_val": floors_h}
    tgt_h = np.zeros_like(H["mask"])
    li = np.flatnonzero(~hid)
    tgt_h[li, H["act"][li]] = 1
    tgt_h[hid] = H["moveset"][hid]
    with inference_globals(2):
        ps = load_checkpoint(PSTAR)
        lgs = logits_and_values(ps, S["rows"], S["mask"].astype(np.uint8))[:, :11]
        p = _softmax(lgs, S["mask"].astype(bool))
        out["selfplay_val"] = {"pstar_entropy": float(ent(p).mean()),
                               "pstar_nll_of_recorded_label": float(-np.log(p[np.arange(len(p)), S["act"]]).mean())}
        del ps
        st = {}
        for lab in ("s_h_5901", "s_p_5901", "s_p0_5901", "s_s_5901"):
            zp = next((STUDENTS / lab / "model").glob("*.zip"))
            m = load_checkpoint(zp)
            lh = logits_and_values(m, H["rows"], H["mask"].astype(np.uint8))[:, :11]
            nh = _set_nll(lh, legal, tgt_h)
            ph = _softmax(lh, legal)
            st[lab] = {"human_val_human_labels": float(nh.mean()),
                       "human_val_labelled_rows_only": float(nh[~hid].mean()),
                       "human_val_hidden_rows_only": float(nh[hid].mean()),
                       "switch_mass_labelled_rows": float(ph[~hid][:, SW].sum(1).mean()),
                       "mass_outside_consistent_set_labelled_rows": float((ph * (~cons)).sum(1)[~hid].mean())}
            del m
        out["students"] = st
    return out


def main(argv) -> int:
    cmds = {"split": cmd_split, "profile": cmd_profile, "logs": cmd_logs, "nll": cmd_nll, "floors": cmd_floors,
            "pairs": cmd_pairs}
    which = argv[1:] or ["split", "profile"]
    res = json.loads(OUT.read_text()) if OUT.exists() else {"schema": "x59s_review_g_checks_v1"}
    for c in which:
        if c == "floors":
            res.setdefault("nll", {})["human_val"] = cmds[c]()
        else:
            res[c] = cmds[c]()
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(res, indent=1, default=float))
        print(json.dumps({c: res["nll"]["human_val"] if c == "floors" else res[c]}, indent=1, default=float)[:6000], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
