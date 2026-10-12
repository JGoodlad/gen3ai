"""Playouts per state and arm: every legal action x 8 worlds (world j on dice seed j of ``turn_seeds(id, 48)``), the
checkpoint GREEDY on both sides to the end; the oracle O: the true battle on dice seeds 0-7 (spec §13.1 item 5).

    python -m replica.play --cp end2 --arm P --shard 0 --nshards 2 --out <dir>

Durable and resumable: one JSON line per state in ``<out>/<cp>/<arm>/shard_<i>_of_<n>.jsonl`` (fsync'd), a state
already present is skipped. Values are the searched side's: +1 win, -1 loss (a stall forfeit included), 0 tie or
truncated (the pilot's convention, so the referee's strings and these agree in kind).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cp", required=True, choices=("end2", "prod"))
    ap.add_argument("--arm", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="only the first N registered states (a subset run)")
    ap.add_argument("--ids", default="", help="a file of state ids to run (one per line), in place of all")
    a = ap.parse_args(argv)

    from replica import env
    env.setup(a.cp)
    import numpy as np  # noqa: F401

    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.reader import inference_globals, load_checkpoint, policy_logits
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from main.policy_spectrum.truth import MicroBatcher, to_log, turn_seeds
    from utils.rust_env.successors import SearchCore, greedy, play_out

    from replica import builder as B
    from replica import fair

    bank = load_bank(env.ARCHIVE / a.cp / "bank")
    dec = {d["id"]: d for d in bank.decisions}
    ids = list(env.registered_states(a.cp))
    if a.ids:
        want = [x.strip() for x in Path(a.ids).read_text().splitlines() if x.strip()]
        ids = [i for i in ids if i in set(want)]
    if a.limit:
        ids = ids[:a.limit]
    ids = ids[a.shard::a.nshards]
    out_dir = Path(a.out) / a.cp / a.arm
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"shard_{a.shard}_of_{a.nshards}.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
    todo = [i for i in ids if i not in done]
    print(f"[replica] {a.cp} {a.arm} shard {a.shard}/{a.nshards}: {len(done)} done, {len(todo)} to go", flush=True)
    t_all = time.monotonic()
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
        with open(out, "a") as f:
            for k, sid in enumerate(todo):
                t0, c0 = time.monotonic(), time.process_time()
                d = dec[sid]
                b = bank.battles[bank.battle_index[d["battle"]]]
                seeds = turn_seeds(sid, 48)
                tokens = {str(x): y for x, y in d["tokens"].items()}
                row: dict = {"id": sid, "cp": a.cp, "arm": a.arm, "schema": "hir_replica_rows_v1"}
                if a.arm == "O":
                    from main.policy_spectrum.truth import cmd_index
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
                    prep = B.prepare(bank, dec, sid)
                    rec = fair.build(prep, a.arm)
                    row.update({"build": {k2: v for k2, v in rec.items() if k2 != "worlds"},
                                "pending_dropped": prep.pending is not None, "gender_draws": prep.gender_draws,
                                "seen": {n: {"species": m.species, "moves": m.moves, "item": m.item,
                                             "ability": m.ability} for n, m in prep.seen.mons.items()}})
                    if not rec.get("ok"):
                        row.update({"ok": False, "reason": rec.get("reason")})
                    else:
                        outcomes = {}
                        worlds = []
                        bad = None

                        def one(jw):
                            j, w = jw
                            teams = [None, None]
                            teams[0 if prep.opp == "p1" else 1] = w["team"]
                            teams[0 if prep.side == "p1" else 1] = prep.own_team
                            cmds = [f"CHOOSE {c[0]} {c[1]}" for c in w["cmds"]] + [f"CHOOSE {d['side']} {d['played']}"]
                            log = {"format_id": b.format_id, "seed": prep.world_seed,
                                   "names": [b.p1["name"], b.p2["name"]], "teams": teams, "cmds": cmds}
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
                row.update({"wall_s": round(time.monotonic() - t0, 3), "cpu_s": round(time.process_time() - c0, 3)})
                f.write(json.dumps(row, sort_keys=True) + "\n")
                f.flush()
                os.fsync(f.fileno())
                if k % 10 == 0 or not row["ok"]:
                    el = time.monotonic() - t_all
                    print(f"[replica] {k + 1}/{len(todo)} {sid} ok={row['ok']} {row.get('reason') or ''} "
                          f"{row['wall_s']}s (elapsed {el / 60:.1f} min)", flush=True)
        pool.shutdown()
        for c in cores:
            c.close()
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.exit(main())
