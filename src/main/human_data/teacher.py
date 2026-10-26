r"""X59-S's two NON-HUMAN label sources, both from frozen P\* (``x59s_matched_students_2026-10-25/PREREG.md``).

    python -m main.human_data.teacher label    --rows <rowset>/rows --pstar <P* zip> --out <rowset>/labels_pstar
    python -m main.human_data.teacher selfplay --pstar <P* zip> --like <rowset>/rows --out <dir>/ss_rows

* ``label`` (the S-P arm): on every row of a human rowset, ONE action sampled from P\*'s policy at T = 1 RESTRICTED
  and renormalised to the row's CONSISTENT set (REVIEW D M2: the legal actions the log leaves possible for the
  human; never a move or bench mon C-GAME drew). The draw is matched noise: its uniform is a pure function of the
  rowset's manifest digest and the row id (:func:`matched_uniform`), so the label never depends on the order, the
  batch or the machine. A hidden-move row gets no P\* sample: both arms train it on the SAME set label (the legal
  moves), so the arms differ there by nothing. P\*'s unrestricted mass outside the consistent set is recorded per
  row (what the restriction removed), with P\*'s NLL of the human's label (a reproduction check against R-b1).
* ``selfplay`` (the S-S arm): P\* against itself on the 719-team pool through the Rust core (the training row path,
  ``utils.rust_env.fixture_battles``' loop with BOTH seats recorded), each decision's action sampled from P\* at
  T = 1 and recorded as the label (on-policy self-play data), until the rowset ``--like`` has as many TRAIN and
  VALIDATION rows and shard sizes; decisions with one legal action are skipped (the human rowset's filter). A game is
  the split unit (its key mod 50 validates). It runs in :data:`SELFPLAY_PARTS` parts, each its own seeded job.

Both write shards in :mod:`.rowset`'s format plus a manifest whose sha256 is the label source's identity. CPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from main.human_data import rowset as RS

SEED_SELFPLAY = 5900
THREADS = 2          # intra-op threads: 6 on the shared agent cores oversubscribed them 60x (measured)
BATCH = 512


def sha256_file(p: Path) -> str:
    return RS.S_sha(p)


def matched_uniform(rowset_sha: str, rid: np.ndarray) -> np.ndarray:
    """The matched-noise uniform of each row: ``sha256(rowset sha | row id)`` → [0, 1) (53 bits)."""
    out = np.empty(len(rid), np.float64)
    for i, r in enumerate(np.asarray(rid, np.int64).tolist()):
        h = hashlib.sha256(f"x59s_pstar_label|{rowset_sha}|{r}".encode()).digest()
        out[i] = (int.from_bytes(h[:8], "big") >> 11) / float(1 << 53)
    return out


def masked_probs(logits: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """Softmax of ``logits`` over ``allowed`` (float64), zero elsewhere."""
    lg = np.where(allowed.astype(bool), logits.astype(np.float64), -np.inf)
    mx = lg.max(1, keepdims=True)
    e = np.where(allowed.astype(bool), np.exp(lg - mx), 0.0)
    return e / e.sum(1, keepdims=True)


def inverse_cdf(probs: np.ndarray, u: np.ndarray) -> np.ndarray:
    """The action whose cumulative probability (index order) first exceeds ``u``; never a zero-probability one."""
    cdf = np.cumsum(probs, 1)
    a = (cdf <= u[:, None]).sum(1)
    a = np.minimum(a, probs.shape[1] - 1)
    bad = probs[np.arange(len(a)), a] <= 0
    if bad.any():                                    # u in the float tail past the last positive bin
        last = probs.shape[1] - 1 - np.argmax(probs[:, ::-1] > 0, axis=1)
        a = np.where(bad, last, a)
    return a.astype(np.int8)


def _load(pstar: Path, threads: int):
    from main.policy_spectrum.reader import load_checkpoint

    return load_checkpoint(Path(pstar))


def _logits(model, rows: np.ndarray, mask: np.ndarray) -> np.ndarray:
    from main.policy_spectrum.reader import policy_logits

    if len(rows) == 0:
        return np.zeros((0, RS.N_ACT), np.float32)
    return policy_logits(model, rows, mask.astype(np.uint8), batch=BATCH)


# ------------------------------------------------------------------------------------------------ S-P labels

def label(rows: Path, pstar: Path, out: Path, threads: int = THREADS) -> Dict[str, Any]:
    """The S-P label column of every shard (module docstring); resumable per shard."""
    from main.policy_spectrum.reader import inference_globals

    man, man_sha = RS.load_manifest(rows)
    out.mkdir(parents=True, exist_ok=True)
    recs = list(man["train_shards"]) + [man["val"]]
    done: List[Dict[str, Any]] = []
    with inference_globals(threads):
        model = _load(pstar, threads)
        for rec in recs:
            p = out / f"{rec['name']}.npz"
            js = out / f"{rec['name']}.json"
            if js.exists():
                done.append(json.loads(js.read_text()))
                continue
            z = RS.load_shard(rows, rec, ("rows", "mask", "act", "hidden", "cons", "rid"))
            lg = _logits(model, z["rows"], z["mask"])
            full = masked_probs(lg, z["mask"])
            pc = masked_probs(lg, z["cons"])
            u = matched_uniform(man_sha, z["rid"])
            act_p = inverse_cdf(pc, u)
            act_p[z["hidden"]] = -1
            out_mass = 1.0 - (full * z["cons"]).sum(1)
            lab = z["act"] >= 0
            if not len(lab):
                lab = np.zeros(0, bool)
            nll_h = np.full(len(lab), np.nan)
            nll_h[lab] = -np.log(np.maximum(full[np.nonzero(lab)[0], z["act"][lab]], 1e-300))
            if (z["cons"][np.nonzero(~z["hidden"])[0], act_p[~z["hidden"]]] != 1).any():
                raise RS.RowsetError(f"{rec['name']}: a P* label outside the consistent set")
            tmp = out / f"{rec['name']}.tmp.npz"
            np.savez_compressed(tmp, act_p=act_p, rid=z["rid"], logits=lg.astype(np.float32),  # type: ignore[arg-type]
                                out_mass=out_mass.astype(np.float32))
            os.replace(tmp, p)
            r = {"name": rec["name"], "rows_sha256": rec["sha256"], "sha256": sha256_file(p), "rows": int(len(lab)),
                 "pstar_out_of_consistent_mass_mean": float(out_mass.mean()),
                 "pstar_nll_of_human_label_mean": float(np.nanmean(nll_h)) if lab.any() else None,
                 "pstar_top1_agrees_with_human": float((lg[lab].argmax(1) == z["act"][lab]).mean()) if lab.any()
                 else None,
                 "sampled_equals_human": float((act_p[lab] == z["act"][lab]).mean()) if lab.any() else None}
            js.write_text(json.dumps(r, indent=1))
            done.append(r)
            print(f"[teacher] {rec['name']}: {r['rows']} rows, out-of-consistent mass "
                  f"{r['pstar_out_of_consistent_mass_mean']:.3f}", flush=True)
    lman = {"schema": "gen3_x59s_pstar_labels_v1", "rows_manifest_sha256": man_sha, "rowset": man["rowset"],
            "pstar": str(pstar), "pstar_sha256": sha256_file(pstar),
            "rule": "one action from P* at T = 1 restricted to the row's consistent set; uniform = "
                    "sha256('x59s_pstar_label|' + rows manifest sha256 + '|' + row id); hidden rows: -1 (the set "
                    "label, as S-H)", "shards": done}
    (out / "labels_manifest.json").write_text(json.dumps(lman, indent=1))
    return lman


# ------------------------------------------------------------------------------------------------ S-S rows

#: the self-play generator runs in PARTS (each its own seed, its own <= 45-min job, resumable at part granularity);
#: ``merge`` concatenates them in part order and cuts the human rowset's exact shard sizes
SELFPLAY_PARTS = 8


def play_part(pstar: Path, want_rows: int, out_npz: Path, seed: int, part: int, threads: int = THREADS,
              n_envs: int = 32) -> Dict[str, Any]:
    """One self-play PART: P* vs P* on the pool until ``want_rows`` rows (both seats, n_legal >= 2) of FINISHED games
    are kept; the games in flight at the end are discarded. Seeds ``[seed, part, 0]`` (teams, battle seeds) and
    ``[seed, part, 1]`` (actions)."""
    from main.policy_spectrum.reader import inference_globals
    from utils.rust_env import ffi
    from utils.rust_env import protocol as P
    from utils.rust_env.episode import stall_threshold
    from utils.rust_env.fixture_battles import NAMES, _lib
    from utils.team_sources import packed_teams

    teams = list(packed_teams("pool"))
    limit = stall_threshold()
    spec = P.spec_json(n=n_envs, threads=1, teams=teams, names=NAMES, turn_limit=limit, refusal_budget=4,
                       bank_dir=None, opponents=[{"kind": "external"}])
    rng_env = np.random.default_rng([seed, part, 0])
    rng_act = np.random.default_rng([seed, part, 1])
    nxt = 0
    staged: Dict[int, int] = {}
    in_flight: Dict[int, int] = {}
    games: List[Dict[str, np.ndarray]] = []
    n_rows = 0
    t0 = time.time()
    with inference_globals(threads):
        model = _load(pstar, threads)
        core = ffi.FfiCore(spec, lib=_lib("release"))
        c = core.cols

        def stage(envs) -> None:
            """The NEXT episode of each env (the core reads it at every auto-reset: kept one episode ahead)."""
            nonlocal nxt
            for e in envs:
                c["ep_team"][e] = rng_env.choice(len(teams), 2, replace=False)
                c["ep_seed"][e] = rng_env.integers(0, 65536, 4)
                c["ep_opp"][e] = 0
                staged[e] = nxt
                nxt += 1

        stage(range(n_envs))
        core.reset()
        in_flight.update(staged)
        stage(range(n_envs))
        rows: Dict[int, List] = {e: [] for e in range(n_envs)}
        try:
            while n_rows < want_rows:
                pend = [(e, s) for e in range(n_envs) for s in (0, 1)
                        if c["need"][e, s] and not (s == 0 and int(c["turn"][e]) >= limit)]
                if pend:
                    obs = np.stack([c["obs"][e, s].copy() for e, s in pend]).astype(np.float32)
                    mask = np.stack([c["mask"][e, s].astype(np.int8) for e, s in pend])
                    pr = masked_probs(_logits(model, obs, mask), mask)
                    acts = inverse_cdf(pr, rng_act.random(len(pend)))
                    for j, (e, s) in enumerate(pend):
                        c["action"][e, s] = int(acts[j])
                        if int(mask[j].sum()) >= 2:
                            rows[e].append((obs[j], mask[j], int(acts[j]), s, int(c["turn"][e])))
                core.step()
                for f in core.finished():
                    e = int(f["env"])
                    got, rows[e] = rows[e], []
                    gid = in_flight[e]
                    in_flight[e] = staged[e]
                    stage([e])
                    if not got or n_rows >= want_rows:
                        continue
                    gk = RS.key64(f"x59s_ss|{seed}|{part}|{gid}")
                    k = len(got)
                    games.append({"rows": np.stack([r[0] for r in got]), "mask": np.stack([r[1] for r in got]),
                                  "act": np.asarray([r[2] for r in got], np.int8),
                                  "seat": np.asarray([r[3] for r in got], np.int8),
                                  "turn": np.asarray([r[4] for r in got], np.int16),
                                  "bkey": np.full(k, gk, np.int64)})
                    n_rows += k
                    if len(games) % 200 == 0:
                        print(f"[teacher] selfplay part {part}: {n_rows}/{want_rows} rows, {len(games)} games "
                              f"({time.time() - t0:.0f} s)", flush=True)
        finally:
            close = getattr(core, "close", None)
            if close is not None:
                close()
    arr = {f: np.concatenate([g[f] for g in games]) for f in ("rows", "mask", "act", "seat", "turn", "bkey")}
    tmp = out_npz.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arr)  # type: ignore[arg-type]
    os.replace(tmp, out_npz)
    rec = {"part": part, "seed": seed, "rows": int(len(arr["act"])), "games": len(games), "games_started": nxt,
           "sha256": sha256_file(out_npz), "seconds": round(time.time() - t0, 1), "threads": threads,
           "n_envs": n_envs}
    out_npz.with_suffix(".json").write_text(json.dumps(rec, indent=1))
    return rec


def _ss_fields(z: Dict[str, np.ndarray], seed: int) -> Dict[str, np.ndarray]:
    """A part's rows in :mod:`.rowset`'s format (known labels; consistent = legal; no hidden rows)."""
    k = len(z["act"])
    kk = np.zeros(k, np.int16)
    starts = np.r_[0, np.nonzero(np.diff(z["bkey"]))[0] + 1, k]
    for a, b in zip(starts[:-1], starts[1:]):
        kk[a:b] = np.arange(b - a)
    return {"rows": z["rows"], "mask": z["mask"], "act": z["act"], "hidden": np.zeros(k, bool), "cons": z["mask"],
            "moveset": np.zeros((k, RS.N_ACT), np.int8),
            "rid": np.asarray([RS.key64(f"x59s_ss|{seed}|{int(b)}|{int(i)}") for b, i in zip(z["bkey"], kk)],
                              np.int64),
            "bkey": z["bkey"], "acct": np.full(k, -1, np.int64), "seat": z["seat"], "k": kk, "turn": z["turn"],
            "forced": np.zeros(k, bool), "rating": np.full(k, -1, np.int16), "quit_dist": np.full(k, -1, np.int16),
            "end": np.zeros(k, np.int8)}


def selfplay(pstar: Path, like: Path, out: Path, threads: int = THREADS, n_envs: int = 32,
             seed: int = SEED_SELFPLAY, parts: int = SELFPLAY_PARTS, only_part: Optional[int] = None,
             max_rows: Optional[int] = None) -> Dict[str, Any]:
    """The S-S rows (module docstring): play every missing PART (or only ``only_part``), then, once all exist, merge
    them into shards of exactly the human rowset's train / validation counts and shard sizes. ``max_rows`` caps the
    TRAIN rows (a smoke)."""
    like_man, like_sha = RS.load_manifest(like)
    want_train = int(like_man["train_rows"]) if max_rows is None else int(max_rows)
    want_val = int(like_man["val_rows"]) if max_rows is None else max(1, int(max_rows) // 50)
    # a part plays its share of (train + val) rows plus 6 % slack for the game-level split's noise
    per = int(np.ceil((want_train + want_val) * 1.06 / parts))
    pdir = out / "parts"
    pdir.mkdir(parents=True, exist_ok=True)
    for i in range(parts):
        if only_part is not None and i != only_part:
            continue
        if not (pdir / f"part_{i:02d}.json").exists():
            r = play_part(pstar, per, pdir / f"part_{i:02d}.npz", seed, i, threads, n_envs)
            print(f"[teacher] selfplay part {i}: {r['rows']} rows in {r['seconds']} s", flush=True)
    if any(not (pdir / f"part_{i:02d}.json").exists() for i in range(parts)):
        return {"done": False}
    # VALIDATION TOP-UP: the human rowset's account split can hold more than 2 % of its rows (hr_spec_v1.r1: 2.36 %),
    # while a game split gives ≈ 2 %; extra parts (seeds [seed, i >= parts, ·]) are played until the validation rows
    # suffice. Their TRAIN rows come after every regular part's, so the train shards fill from the regular parts first.
    def _val_rows(i: int) -> int:
        bk = np.load(pdir / f"part_{i:02d}.npz")["bkey"]
        return int(sum(RS.is_val_key(b) for b in bk))

    n_parts = parts
    have_val = sum(_val_rows(i) for i in range(parts))
    while have_val < want_val:
        if not (pdir / f"part_{n_parts:02d}.json").exists():
            need = int(np.ceil((want_val - have_val) * 50 * 1.5))
            r = play_part(pstar, need, pdir / f"part_{n_parts:02d}.npz", seed, n_parts, threads, n_envs)
            print(f"[teacher] selfplay validation top-up part {n_parts}: {r['rows']} rows", flush=True)
        have_val += _val_rows(n_parts)
        n_parts += 1
    sizes = [int(s["rows"]) for s in like_man["train_shards"]] if max_rows is None else \
        [min(RS.SHARD, want_train - i) for i in range(0, want_train, RS.SHARD)]
    # STREAMED: one part in memory at a time plus the carry (the rows of 1.3M decisions do not fit at once)
    shards: List[Dict[str, Any]] = []
    carry: List[Dict[str, np.ndarray]] = []
    n_carry = 0
    val: List[Dict[str, np.ndarray]] = []
    n_val = 0
    recs = []
    for i in range(n_parts):
        rec = json.loads((pdir / f"part_{i:02d}.json").read_text())
        if sha256_file(pdir / f"part_{i:02d}.npz") != rec["sha256"]:
            raise RS.RowsetError(f"self-play part {i}: sha256 differs from its record")
        recs.append(rec)
        z = _ss_fields(dict(np.load(pdir / f"part_{i:02d}.npz")), seed)
        v = np.array([RS.is_val_key(b) for b in z["bkey"]], bool)
        if n_val < want_val:
            val.append({f: z[f][v] for f in RS.FIELDS})
            n_val += int(v.sum())
        carry.append({f: z[f][~v] for f in RS.FIELDS})
        n_carry += int((~v).sum())
        del z
        while len(shards) < len(sizes) and n_carry >= sizes[len(shards)]:
            cat = {f: np.concatenate([t[f] for t in carry]) for f in RS.FIELDS}
            n = sizes[len(shards)]
            shards.append(RS._write_shard(out, f"train_{len(shards):04d}", {f: cat[f][:n] for f in RS.FIELDS}))
            carry = [{f: cat[f][n:] for f in RS.FIELDS}]
            n_carry -= n
            del cat
    if len(shards) < len(sizes) or n_val < want_val:
        raise RS.RowsetError(f"the parts hold too few rows ({len(shards)} / {len(sizes)} shards, {n_val} / {want_val} "
                             "validation rows): raise the slack")
    vcat = {f: np.concatenate([t[f] for t in val]) for f in RS.FIELDS}
    vrec = RS._write_shard(out, "val", {f: vcat[f][:want_val] for f in RS.FIELDS})
    man = {"schema": "gen3_x59s_selfplay_rows_v1", "rowset": f"ss_like_{like_man['rowset']}",
           "like_rows_manifest_sha256": like_sha, "pstar": str(pstar), "pstar_sha256": sha256_file(pstar),
           "seed": seed, "parts": recs, "teams": "the 719-team pool (utils.team_sources.packed_teams('pool'))",
           "rule": "P* vs P*, both seats recorded, action ~ P* at T = 1 (the label); n_legal >= 2; finished games "
                   "only; parts concatenated in order; game key mod 50 validates; cut to the human rowset's shard "
                   "sizes", "train_shards": shards, "val": vrec, "train_rows": int(sum(s["rows"] for s in shards)),
           "val_rows": vrec["rows"], "train_hidden": 0, "code": RS._code_identity()}
    (out / "rows_manifest.json").write_text(json.dumps(man, indent=1))
    return {"done": True, **man}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.human_data.teacher", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    lb = sub.add_parser("label")
    lb.add_argument("--rows", required=True)
    sp = sub.add_parser("selfplay")
    sp.add_argument("--like", required=True)
    sp.add_argument("--max-rows", type=int, default=None)
    sp.add_argument("--envs", type=int, default=32)
    sp.add_argument("--part", type=int, default=None, help="play only this part (each part is its own job)")
    for p in (lb, sp):
        p.add_argument("--pstar", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--threads", type=int, default=THREADS)
    a = ap.parse_args(argv)
    try:
        RS._guard_out(Path(a.out))
        if a.cmd == "label":
            m = label(Path(a.rows), Path(a.pstar), Path(a.out), a.threads)
            print(json.dumps({"shards": len(m["shards"])}), flush=True)
        else:
            m = selfplay(Path(a.pstar), Path(a.like), Path(a.out), a.threads, a.envs, only_part=a.part,
                         max_rows=a.max_rows)
            print(json.dumps({k: m[k] for k in ("done", "train_rows", "val_rows") if k in m}), flush=True)
    except RS.RowsetError as exc:
        print(f"[teacher] REFUSED: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
