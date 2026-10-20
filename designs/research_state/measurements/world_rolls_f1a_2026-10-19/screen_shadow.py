"""F1 tier A, STEP 1 (measure before building the builder option): of the candidates TODAY's honest screen replays
and REJECTS, how many would a 16-roll damage check accept? (``designs/endstate/design_world_sampler.md`` §3.3.1.)

Each state is built by TODAY's builder (``WorldSampler(reveal="off")``, J 8, ``max_tries`` 60, the studies' call). Every
gate replay it runs is SHADOWED: the same battle is replayed again with the Rust core's damage-roll control in
``match`` mode (``pokesim::rolls``: every damage roll's VALUE may be any roll that prints what the searched side saw,
the dice stream kept aligned), against the targets the TRUTH's own replay recorded. Both verdicts use the builder's
gate (the searched side's rows AND protocol byte-equal to the truth's). Today's verdicts alone drive the builder, so
the shadow counts are of the proposals today's chain made (a LOWER bound on what a tier-A chain would find: it would
have moved on the wider verdicts). World BUILDS only: no playout, no model, no outcome.

The invariant the shadow checks on every replay: a candidate today ACCEPTS is accepted by tier A too (the control
prefers the stream's roll whenever it is consistent, so today's trajectory is reproduced). A violation is a DEFECT
and is counted (``today_only``), never silently dropped.

States: a seeded random prefix of the stage-0 population (``--n-random``) plus a seeded sample of the states the
stored F rows EXCLUDED (``--n-excluded``), read-only from the archives (the F2 read's ``populations``).

    PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
    CUDA_VISIBLE_DEVICES="" POKESIM_EMISSION_SELFCHECK=1 nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh 6 \\
        $PY -I screen_shadow.py --shard 0 --of 2 --out <dir>

One row per state appended to ``<out>/shadow_<shard>_of_<W>.jsonl`` (resumable).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE.parent / "world_legality_f2_2026-10-19"))

J = 8
MAX_TRIES = 60


def sample_ids(stored: Dict[str, dict], n_random: int, n_excluded: int) -> List[str]:
    order = sorted(stored, key=lambda i: hashlib.sha256(f"f1a_2026-10-19:{i}".encode()).hexdigest())
    rand = order[:n_random]
    excl = [i for i in order if not stored[i].get("ok") and i not in rand][:n_excluded]
    return rand + excl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-random", type=int, default=60)
    ap.add_argument("--n-excluded", type=int, default=30)
    a = ap.parse_args()

    import main as _m

    if not str(Path(_m.__file__).resolve()).startswith(str(REPO / "src")):
        raise SystemExit(f"main imported from {_m.__file__}, not this checkout: run with -I")
    import rebuild_worlds as F2                      # the F2 read's population readers (read only)
    from agents.battle import core_replay as CR
    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.truth import cmd_index, to_log
    from main.value_search.worlds import WorldError, WorldSampler, prepare_root, protocol_lines, reveals

    orig = CR.run_core
    ctx: dict = {}

    def verdict(res: dict) -> bool:
        if not res["ok"]:
            return False
        try:
            _, rows = reveals(res, ctx["viewer"], ctx["n"])
        except WorldError:
            return False
        return rows == ctx["ref"] and protocol_lines(res, ctx["viewer"], ctx["n"]) == ctx["ref_proto"]

    def shadow_run_core(battles, *args, **kw):
        out = orig(battles, *args, **kw)
        if "targets" not in ctx or not battles or "#s" not in battles[0].label:
            return out
        roll = {"viewer": ctx["viewer"], "mode": "match", "targets": ctx["targets"]}
        tier_a = orig([dataclasses.replace(b, rolls=roll) for b in battles], *args, **kw)
        for b, r0, r1 in zip(battles, out, tier_a):
            kind = "screen" if any(len(c) > 2 for c in b.commands) else "joint"
            v0, v1 = verdict(r0), verdict(r1)
            c = ctx["counts"].setdefault(kind, {"n": 0, "today": 0, "tier_a": 0, "rescued": 0, "today_only": 0})
            c["n"] += 1
            c["today"] += v0
            c["tier_a"] += v1
            c["rescued"] += (v1 and not v0)
            c["today_only"] += (v0 and not v1)
            if v1:
                rr = r1["rolls"]
                ctx["log_w"].append(rr["log_w"])
                if not v0:
                    ctx["widened_sites"].append(len(rr["force"]))
        return out

    CR.run_core = shadow_run_core

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stored = F2.stored_rows()
    ids = sample_ids(stored, a.n_random, a.n_excluded)
    rand = set(ids[: a.n_random])
    ids = [i for k, i in enumerate(ids) if k % a.of == a.shard]
    path = out / f"shadow_{a.shard}_of_{a.of}.jsonl"
    done = {json.loads(x)["id"] for x in path.read_text().splitlines() if x.strip()} if path.exists() else set()
    ids = [i for i in ids if i not in done]
    sampler = WorldSampler(reveal="off", max_tries=MAX_TRIES)
    banks = {pop: load_bank(bank) for pop, bank, _f, _k in F2.populations()}
    decs = {pop: {d["id"]: d for d in b.decisions} for pop, b in banks.items()}
    head = os.popen(f"git -C {REPO} rev-parse HEAD").read().strip()
    print(f"[f1a shadow] shard {a.shard}/{a.of}: {len(ids)} states at {head[:8]}", flush=True)
    t0 = time.monotonic()
    with open(path, "a") as fo:
        for n_i, did in enumerate(ids):
            pop = stored[did]["pop"]
            bank = banks[pop]
            d = decs[pop][did]
            b = bank.battles[bank.battle_index[d["battle"]]]
            side = d["side"]
            at = cmd_index(b, side, d["n"])
            log = to_log(b)
            row = {"id": did, "pop": pop, "head": head, "stratum": "random" if did in rand else "excluded",
                   "stored_ok": bool(stored[did].get("ok"))}
            w0 = time.monotonic()
            ctx.clear()
            try:
                root = prepare_root(log, at, side, did)
                rec = orig([dataclasses.replace(root.rec(None, f"{did}#truth", root.cmds),
                                                rolls={"viewer": root.viewer, "mode": "record"})],
                           trackers=True, obs=True)[0]
                tg = rec["rolls"]["targets"]
                ctx.update(viewer=root.viewer, n=root.n, ref=root.ref, ref_proto=root.ref_proto, targets=tg,
                           counts={}, log_w=[], widened_sites=[])
                row["sites"] = len(tg)
                row["pinned_sites"] = sum(1 for t in tg if t[5])
                row["site_kinds"] = sorted({t[0] for t in tg})
                W = sampler.sample(log, at, side, J, key=did, screen_cache={})
                row.update(ok=True, fails=W.fails, screens=W.screens)
            except WorldError as exc:
                row.update(ok=False, error=f"{type(exc).__name__}: {str(exc)[:400]}")
            except Exception as exc:
                row.update(ok=False, crash=True, error=f"{type(exc).__name__}: {str(exc)[:400]}",
                           trace=traceback.format_exc()[-1500:])
                print(f"[f1a shadow] DEFECT {did}: {row['error'][:300]}", flush=True)
            row["counts"] = ctx.get("counts", {})
            row["log_w"] = ctx.get("log_w", [])[:200]
            row["widened_sites"] = ctx.get("widened_sites", [])
            row["wall_s"] = round(time.monotonic() - w0, 2)
            fo.write(json.dumps(row, sort_keys=True) + "\n")
            fo.flush()
            print(f"[f1a shadow] {n_i + 1}/{len(ids)} {did} ok={row['ok']} {row['counts']} "
                  f"{(time.monotonic() - t0) / 60:.1f} min", flush=True)
    print("[f1a shadow] done", flush=True)


if __name__ == "__main__":
    main()
