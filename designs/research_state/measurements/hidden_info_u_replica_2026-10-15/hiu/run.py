"""The resumable runner: per registered state, every arm's worlds (built, guarded) and playouts.

    python -m hiu.run --cp end2 --out ~/gen3ai_archive/hidden_info_u_replica [--ids FILE] [--limit N]

Per state, in order: O, F, F+U, F+M, O-M, O-U (``ARMS``). Each arm writes ONE JSON line to
``<out>/rows/<cp>/<arm>/w<pid>.jsonl`` (fsync'd); a state whose every arm has a row gets a done-marker
``<out>/done/<cp>/<sha1(id)>`` holding the id. Any number of workers may run per checkpoint: a worker CLAIMS a
state by creating ``<out>/claims/<cp>/<sha1(id)>`` exclusively (its pid inside) and skips a state claimed by a live
pid; a dead worker's claim is taken over, and the taker re-reads that state's rows (the arms already written are
skipped; F+U re-reads F's worlds from F's row). States run in the order sha256("hiu_u_replica_2026:" + id), so the
done set is close to a uniform random subset of the 600 at any time.

Playouts follow the replica's ``replica.play`` exactly: every legal action x 8 worlds, world j on dice seed j of
``turn_seeds(id, 48)``, the checkpoint GREEDY on both sides to the end (stall rule "production", 999 turns); O: the
true battle on dice seeds 0-7. Values are the searched side's: +1 win, -1 loss, 0 tie or truncation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ARMS = ("O", "F", "F+U", "F+M", "O-M", "O-U")
SCHEMA = "hiu_rows_v1"


def order_key(sid: str) -> str:
    return hashlib.sha256(("hiu_u_replica_2026:" + sid).encode()).hexdigest()


def done_path(out: Path, cp: str, sid: str) -> Path:
    return out / "done" / cp / hashlib.sha1(sid.encode()).hexdigest()


def load_rows(arm_dir: Path, only: str = "") -> dict:
    """Every row of one (cp, arm) over all workers' files (``only``: one state's rows)."""
    rows: dict = {}
    for p in sorted(arm_dir.glob("*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip() and (not only or only in line):
                r = json.loads(line)
                if not only or r["id"] == only:
                    rows.setdefault(r["id"], r)
    return rows


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def claim(path: Path) -> str:
    """'new' (claimed now), 'stale' (a dead worker's claim, taken over) or 'busy' (a live worker holds it)."""
    me = f"{os.getpid()} {time.time():.0f}\n"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, me.encode())
        os.close(fd)
        return "new"
    except FileExistsError:
        pass
    try:
        pid = int(path.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        pid = -1
    if pid == os.getpid():
        return "new"
    if pid > 0 and _alive(pid):
        return "busy"
    try:                           # atomic takeover: only one worker's rename of the dead claim succeeds
        os.rename(path, path.parent / f"{path.name}.dead{pid}.by{os.getpid()}")
    except FileNotFoundError:
        return "busy"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return "busy"
    os.write(fd, me.encode())
    os.close(fd)
    return "stale"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cp", required=True, choices=("end2", "prod"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", default="", help="a file of state ids (one per line): only these")
    ap.add_argument("--limit", type=int, default=0, help="only the first N states in the run order")
    ap.add_argument("--arms", default=",".join(ARMS))
    a = ap.parse_args(argv)
    arms = [x for x in a.arms.split(",") if x]
    if any(x not in ARMS for x in arms):
        raise SystemExit(f"unknown arm in {arms}")

    from replica import env
    env.setup(a.cp)
    from hiu import fix
    ls = fix.install()
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np

    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.reader import inference_globals, load_checkpoint, policy_logits
    from main.policy_spectrum.truth import MicroBatcher, cmd_index, to_log, turn_seeds
    from replica import builder as B
    from utils.rust_env.successors import SearchCore, SuccessorsError, greedy, play_out

    from hiu import arms as A
    from hiu import guard

    out = Path(os.path.expanduser(a.out))
    rows_dir = out / "rows" / a.cp
    arm_dirs = {arm: rows_dir / arm for arm in ARMS}
    for p in list(arm_dirs.values()) + [out / "done" / a.cp, out / "claims" / a.cp]:
        p.mkdir(parents=True, exist_ok=True)
    files = {arm: arm_dirs[arm] / f"w{os.getpid()}.jsonl" for arm in ARMS}
    have = {arm: set(load_rows(arm_dirs[arm])) for arm in ARMS}

    bank = load_bank(env.ARCHIVE / a.cp / "bank")
    dec = {d["id"]: d for d in bank.decisions}
    ids = sorted(env.registered_states(a.cp), key=order_key)
    if a.ids:
        want = {x.strip() for x in Path(a.ids).read_text().splitlines() if x.strip()}
        ids = [i for i in ids if i in want]
    if a.limit:
        ids = ids[:a.limit]
    todo = [i for i in ids if not done_path(out, a.cp, i).exists()]
    print(f"[hiu] {a.cp}: {len(ids) - len(todo)} done, {len(todo)} to go; arms {arms}; learnset {ls.sha256[:12]}",
          flush=True)

    def first_legal(rows, masks, who):
        return [int(np.flatnonzero(m)[0]) for m in masks]

    t_all = time.monotonic()
    handles = {arm: open(files[arm], "a") for arm in ARMS}
    with inference_globals(1):
        model = load_checkpoint(env.checkpoint(a.cp))
        policy = greedy(MicroBatcher(lambda r, m: policy_logits(model, r, m)))
        core = SearchCore(path=env.lib_path(a.cp))
        local = threading.local()
        cores = [core]
        pool = ThreadPoolExecutor(max_workers=8)

        def world_core():
            c = getattr(local, "core", None)
            if c is None:
                c = local.core = SearchCore(path=env.lib_path(a.cp))
                cores.append(c)
            return c

        for k, sid in enumerate(todo):
            if done_path(out, a.cp, sid).exists():
                continue
            cl = claim(out / "claims" / a.cp / done_path(out, a.cp, sid).name)
            if cl == "busy":
                continue
            if cl == "stale":           # a dead worker's state: re-read what it wrote
                for arm in ARMS:
                    if load_rows(arm_dirs[arm], only=sid):
                        have[arm].add(sid)
            d = dec[sid]
            b = bank.battles[bank.battle_index[d["battle"]]]
            seeds = turn_seeds(sid, 48)
            tokens = {str(x): y for x, y in d["tokens"].items()}
            prep = None
            f_worlds = None
            if sid in have["F"]:
                fr = load_rows(arm_dirs["F"], only=sid).get(sid)
                if fr and fr.get("ok"):
                    f_worlds = [w["team"] for w in fr["worlds"]]
            t_state = time.monotonic()
            for arm in arms:
                if sid in have[arm]:
                    continue
                t0, c0 = time.monotonic(), time.process_time()
                row: dict = {"id": sid, "cp": a.cp, "arm": arm, "schema": SCHEMA, "learnset_sha256": ls.sha256}
                try:
                    if arm == "O":
                        at = cmd_index(b, d["side"], d["n"])
                        res = play_out(to_log(b), at, d["side"], policy=policy, seeds=seeds[:8], core=core,
                                       stall="production", max_turns=999)
                        got = {str(x): y for x, y in sorted(res.tokens.items())}
                        outcomes: dict = {}
                        for br in sorted(res.branches, key=lambda x: (int(x["action"]), int(x["seed"]))):
                            outcomes.setdefault(str(int(br["action"])), []).append(res.value(br))
                        row.update({"ok": got == tokens, "reason": None if got == tokens else f"tokens {got}",
                                    "outcomes": outcomes, "other_open": res.root.get("other_open")})
                    else:
                        if prep is None:
                            prep = B.prepare(bank, dec, sid)
                            if not prep.error:
                                # the guard's self-test: the TRUE team through the same engine check
                                guard.check(prep, "ID", ["]".join(guard.true_entry(prep, i) for i in range(6))])
                        pr = prep

                        def validate(team, cmds_, pr=pr):
                            teams_ = [None, None]
                            teams_[0 if pr.opp == "p1" else 1] = team
                            teams_[0 if pr.side == "p1" else 1] = pr.own_team
                            log_ = {"format_id": b.format_id, "seed": pr.world_seed,
                                    "names": [b.p1["name"], b.p2["name"]], "teams": teams_,
                                    "cmds": [f"CHOOSE {c[0]} {c[1]}" for c in cmds_] + [f"CHOOSE {d['side']} {d['played']}"]}
                            try:
                                play_out(log_, len(cmds_), d["side"], policy=first_legal, seeds=[seeds[0]],
                                         actions=[int(next(iter(d["tokens"])))], core=core, max_turns=1, stall=None)
                                return True
                            except SuccessorsError:
                                return False

                        if arm in ("F", "F+M"):
                            rec = A.build_fair(prep, arm, validate)
                        elif arm == "F+U":
                            rec = A.build_fpu(prep, f_worlds, validate)
                        else:
                            rec = A.build_true_family(prep, arm, validate)
                        row.update({"build": {k2: v for k2, v in rec.items() if k2 != "worlds"},
                                    "pending": prep.pending is not None, "gender_draws": prep.gender_draws,
                                    "k_seen": len(prep.seen.mons), "u_slots": guard.u_slots(prep),
                                    "seen": {n: {"species": m.species, "moves": m.moves, "item": m.item,
                                                 "ability": m.ability, "slot": prep.seen_slot.get(n)}
                                             for n, m in prep.seen.mons.items()}})
                        if not rec.get("ok"):
                            row.update({"ok": False, "reason": rec.get("reason")})
                        else:
                            teams = [w["team"] for w in rec["worlds"]]
                            row["guard"] = guard.check(prep, arm, teams, f_worlds if arm == "F+U" else None)
                            outcomes = {}
                            worlds = []
                            bad = None

                            def one(jw, pr=pr):
                                j, w = jw
                                teams_ = [None, None]
                                teams_[0 if pr.opp == "p1" else 1] = w["team"]
                                teams_[0 if pr.side == "p1" else 1] = pr.own_team
                                cmds = [f"CHOOSE {c[0]} {c[1]}" for c in w["cmds"]] + [f"CHOOSE {d['side']} {d['played']}"]
                                log = {"format_id": b.format_id, "seed": pr.world_seed,
                                       "names": [b.p1["name"], b.p2["name"]], "teams": teams_, "cmds": cmds}
                                return play_out(log, len(w["cmds"]), d["side"], policy=policy, seeds=[seeds[j]],
                                                core=world_core(), stall="production", max_turns=999)

                            results = list(pool.map(one, list(enumerate(rec["worlds"]))))
                            for j, (w, res) in enumerate(zip(rec["worlds"], results)):
                                got = {str(x): y for x, y in sorted(res.tokens.items())}
                                if got != tokens:
                                    bad = f"world {j}: root tokens {got} != {tokens}"
                                    break
                                for br in res.branches:
                                    outcomes.setdefault(str(int(br["action"])), []).append(res.value(br))
                                worlds.append({"team": w["team"], "rendered": w["rendered"], "origin": w["origin"],
                                               "other_open": res.root.get("other_open")})
                            row.update({"ok": bad is None, "reason": bad, "outcomes": outcomes if bad is None else {},
                                        "worlds": worlds})
                            if arm == "F":
                                f_worlds = teams if bad is None else None
                except guard.GuardError as exc:
                    row.update({"ok": False, "reason": f"GUARD: {str(exc)[:400]}"})
                    print(f"[hiu] GUARD RAISED {sid} {arm}: {exc}", flush=True)
                except Exception as exc:          # a state the builder or the core refuses: a FAILED row
                    import traceback
                    row.update({"ok": False, "reason": f"exception: {type(exc).__name__}: {str(exc)[:200]}",
                                "trace": traceback.format_exc()[-1500:]})
                    core = SearchCore(path=env.lib_path(a.cp))      # a Rust panic poisons a handle
                    cores.append(core)
                    local = threading.local()
                    if arm == "F":
                        f_worlds = None
                row.update({"wall_s": round(time.monotonic() - t0, 3), "cpu_s": round(time.process_time() - c0, 3)})
                fh = handles[arm]
                fh.write(json.dumps(row, sort_keys=True) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
                have[arm].add(sid)
                if not row["ok"]:
                    print(f"[hiu]   {sid} {arm} FAILED: {row.get('reason')}", flush=True)
            if all(sid in have[x] for x in ARMS):
                done_path(out, a.cp, sid).write_text(sid + "\n")
            el = time.monotonic() - t_all
            print(f"[hiu] {k + 1}/{len(todo)} {sid} {time.monotonic() - t_state:.0f}s (elapsed {el / 60:.1f} min; "
                  f"guard {dict(guard.COUNTS)})", flush=True)
        pool.shutdown()
        for c in cores:
            c.close()
    for fh in handles.values():
        fh.close()
    print(f"[hiu] FINISHED {a.cp}: guard {dict(guard.COUNTS)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.exit(main())
