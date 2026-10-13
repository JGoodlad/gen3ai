"""TIER 1, step 1: the WORLDS of every arm at every state, built at THIS STUDY'S HEAD (no model, no playout, no outcome).

    run_head.sh 8 24h build_worlds.py --cp end2 --states registered [--arms F,F+M,...] [--shard i --of W] [--limit N]

Per state (the registered 600 or the supplement, `t1_common.bank_path`) and arm (`t1_common.ARMS`, in `BUILD_ORDER`):
8 gate-passed worlds from the information-lever builder (`main.value_search.worlds.WorldSampler(reveal=<spec>,
max_tries=60)`, the registration's ≤ 60 tries, honest by default) or this unit's prior builds (FN, POOLSETS, POOL;
README §4). ``key`` = the decision id for EVERY arm, so the seen-mon screen of one decision is deterministic per
(decision, constraint) and arms with F's constraint (D50, D90, W90, MAP100, F+U, FN) take F's seen-mon draws (common
random numbers for their paired contrasts; one ``screen_cache`` per state also saves the CPU).

One durable row per (state, arm) in ``<archive>/<cp>/worlds/<code>/worlds_<states>_<shard>_of_<W>.jsonl``, in the
pilot's det format (``teams`` = the 8 opponent packed teams, ``cmds`` = each world's command prefix up to the root as
[side, token] pairs; `run_det_playouts.world_log` rebuilds the playout log at the pin) plus the builder's provenance
(`facts`, `leak`, `fails`, `tries`, `screens`), its cost, and ``correct`` = per world the number of correct hidden facts
(the seen mons' true unrevealed moves held + the world's never-seen species that are true never-seen species; §6.9 M5X:
the MOST WRONG world is fixed here, before any outcome exists). A state the builder cannot fit (a ``WorldError``) is a
row with ``ok`` false and the reason: an EXCLUSION, counted. Codes come from the sealed key (`t1_common.load_key`).
"""
from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1_common as T  # noqa: E402


def cpu_seconds() -> float:
    a = resource.getrusage(resource.RUSAGE_SELF)
    b = resource.getrusage(resource.RUSAGE_CHILDREN)
    return a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime


def head_sha() -> str:
    return subprocess.run(["git", "-C", str(T.HEAD_WT), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()


def split_prefix(log: dict, at: int) -> List[List[str]]:
    out = []
    for c in log["cmds"][:at]:
        if not c.startswith("CHOOSE "):
            raise RuntimeError(f"a non-CHOOSE command in a world's root prefix: {c!r}")
        _, side, tok = c.split(" ", 2)
        out.append([side, tok])
    return out


def correct_counts(worlds_teams: List[str], true_team: str, revealed: Dict[str, List[str]]) -> List[int]:
    """Per world: (true unrevealed moves of each SEEN mon the world's set holds, a Hidden Power counted whatever its
    type) + (the world's never-seen species that are true never-seen species)."""
    from main.value_search.world_draws import is_hp, parse_set, toid, typed_moves

    def canon(m: str) -> str:
        return "hiddenpower" if is_hp(m) else m

    def mons(team: str) -> Dict[str, List[str]]:
        out = {}
        for e in team.split("]"):
            f = parse_set(e)
            out[toid(f[1] or f[0])] = [canon(m) for m in typed_moves(f)]
        return out

    tru = mons(true_team)
    seen = set(revealed)
    true_never = {s for s in tru if s not in seen}
    res = []
    for tw in worlds_teams:
        w = mons(tw)
        c = 0
        for s in seen:
            if s not in tru or s not in w:
                continue
            rev = {canon(m) for m in revealed[s]}
            unrev_true = [m for m in tru[s] if m not in rev]
            c += sum(1 for m in unrev_true if m in set(w[s]))
        c += sum(1 for s in w if s not in seen and s in true_never)
        res.append(int(c))
    return res


class Sources:
    """label → a sampler with ``.sample(log, at, side, J, key, screen_cache=...)`` (README §4)."""

    def __init__(self, cp: str, states: str):
        from main.value_search.worlds import WorldSampler

        self.cp, self.states = cp, states
        self._ws = {lab: WorldSampler(reveal=a["spec"], max_tries=60) for lab, a in T.ARMS.items() if a["kind"] == "spec"}
        self._fn_belief: Optional[Dict[str, dict]] = None
        self._poolsets = None
        self._pool = None

    def fn_belief(self) -> Dict[str, dict]:
        if self._fn_belief is None:
            p = T.cp_dir(self.cp) / f"fn_belief_{self.states}.jsonl"
            self._fn_belief = {r["id"]: r for r in T.read_rows(p)}
        return self._fn_belief

    def sample(self, label: str, log: dict, at: int, side: str, did: str, caches: Dict[str, dict]):
        kind = T.ARMS[label]["kind"]
        if kind == "spec":
            return self._ws[label].sample(log, at, side, T.J, key=did, screen_cache=caches["F"])
        if kind == "fn":
            import fn_worlds

            bel = self.fn_belief().get(did)
            if bel is None:
                raise LookupError(f"{did}: no network-belief row (fn_belief_{self.states}.jsonl)")
            return fn_worlds.fn_sampler(bel, max_tries=60).sample(log, at, side, T.J, key=did,
                                                                 screen_cache=caches["F"])
        if kind == "poolsets":
            import poolsets_worlds

            if self._poolsets is None:
                self._poolsets = poolsets_worlds.poolsets_sampler(max_tries=60)
            return self._poolsets.sample(log, at, side, T.J, key=did, screen_cache=caches["POOLSETS"])
        if kind == "pool":
            import pool_worlds

            if self._pool is None:
                self._pool = pool_worlds.PoolSampler(max_tries=60)
            return self._pool.sample(log, at, side, T.J, key=did)
        raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cp", required=True, choices=sorted(T.CPS))
    ap.add_argument("--states", required=True, choices=("registered", "supp"))
    ap.add_argument("--arms", default=None, help="comma list of labels (default: every arm registered for these states)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", default=None, help="a file of decision ids (one per line) to restrict to")
    ap.add_argument("--validation", default=None,
                    help="a VALIDATION build (not an arm): an information-lever spec (e.g. oracle), written under "
                         "code V_<spec>; README §5")
    a = ap.parse_args()
    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.truth import cmd_index, to_log
    from main.value_search.worlds import WorldError

    key = T.load_key()
    if a.validation:
        lab = f"V:{a.validation}"
        T.ARMS[lab] = {"kind": "spec", "spec": a.validation, "states": "all"}
        T.BUILD_ORDER = (lab,)
        key[lab] = "V_" + a.validation.replace("+", "p").replace("-", "m")
        a.arms = lab
    want = [x for x in T.BUILD_ORDER
            if (a.arms is None or x in a.arms.split(",")) and (a.states == "registered" or T.ARMS[x]["states"] == "all")]
    if a.arms is not None and set(a.arms.split(",")) - set(want):
        raise SystemExit(f"arms not registered for {a.states}: {sorted(set(a.arms.split(',')) - set(want))}")
    bank = load_bank(T.bank_path(a.cp, a.states))
    ids = sorted(d["id"] for d in bank.decisions)
    if a.ids:
        keep = {x.strip() for x in Path(a.ids).read_text().splitlines() if x.strip()}
        ids = [x for x in ids if x in keep]
    ids = T.shard(ids, a.shard, a.of)[: a.limit]
    done: Dict[str, set] = {}
    files = {}
    for lab in want:
        d = T.worlds_dir(a.cp, key[lab])
        done[lab] = set(T.read_dir(d, f"worlds_{a.states}_*.jsonl"))
        files[lab] = open(d / f"worlds_{a.states}_{a.shard}_of_{a.of}.jsonl", "a")
    src = Sources(a.cp, a.states)
    sha = head_sha()
    dec = {d["id"]: d for d in bank.decisions}
    print(f"[worlds] {a.cp} {a.states} shard {a.shard}/{a.of}: {len(ids)} states x {want}", flush=True)
    t0 = time.monotonic()
    try:
        for n_i, did in enumerate(ids):
            todo = [lab for lab in want if did not in done[lab]]
            if not todo:
                continue
            d = dec[did]
            b = bank.battles[bank.battle_index[d["battle"]]]
            side = d["side"]
            at = cmd_index(b, side, d["n"])
            log = to_log(b)
            oi = 1 if side == "p1" else 0
            caches: Dict[str, dict] = {"F": {}, "POOLSETS": {}}
            for lab in todo:
                c0, w0 = cpu_seconds(), time.monotonic()
                row = {"id": did, "code": key[lab], "cp": a.cp, "states": a.states, "head": sha, "at": at,
                       "side": side}
                try:
                    W = src.sample(lab, log, at, side, did, caches)
                    teams = [lg["teams"][oi] for lg in W.logs]
                    if teams != list(W.teams):
                        raise RuntimeError("Worlds.teams differ from the logs' opponent teams")
                    row.update({"ok": True, "teams": teams, "cmds": [split_prefix(lg, at) for lg in W.logs],
                                "tries": W.tries, "fails": W.fails, "screens": getattr(W, "screens", 0),
                                "leak": W.leak, "revealed": W.revealed, "spec": W.spec,
                                "facts": W.facts, "samplers": getattr(W, "samplers", []),
                                "correct": correct_counts(teams, log["teams"][oi], W.revealed)})
                    extra = getattr(W, "extra", None)
                    if extra:
                        row["extra"] = extra
                except (WorldError, LookupError) as exc:
                    row.update({"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:600]}"})
                except Exception as exc:                       # a builder crash is a FINDING, never a silent skip
                    row.update({"ok": False, "crash": True,
                                "error": f"{type(exc).__name__}: {str(exc)[:600]}",
                                "trace": traceback.format_exc()[-2000:]})
                row["cost"] = {"cpu_s": round(cpu_seconds() - c0, 3), "wall_s": round(time.monotonic() - w0, 3)}
                files[lab].write(json.dumps(row, sort_keys=True, default=str) + "\n")
                files[lab].flush()
                import os

                os.fsync(files[lab].fileno())
                if not row["ok"]:
                    print(f"[worlds] {lab} EXCLUDED {did}: {row['error'][:200]}", flush=True)
            el = time.monotonic() - t0
            print(f"[worlds] {n_i + 1}/{len(ids)} {el / 60:.1f} min", flush=True)
    finally:
        for f in files.values():
            f.close()


if __name__ == "__main__":
    main()
