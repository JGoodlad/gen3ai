"""E1-0's PLAY side (``PREREGISTRATION.md`` §3): per bot-stratum state of the maturity battery's interim LEAN bank, every
legal action x the referee's 24 seeds (``truth.turn_seeds(id, 48)[16:40]``, the banked referee's own) played to the END
in the TRUE world at X26 225M, by ``main.value_search.playout.referee`` loaded BY FILE PATH from THIS checkout (so the
shipped toggle is the code that runs):

  --mode bot    the opponent is the state's own in-core scripted bot (its name from the state's ``cycle_opp``; route
                seed ``bot_seed(id)``), the forward greedy on our side only;
  --mode self   the SELF-referee (the default, OFF), the forward greedy on both sides: G-REPRO, which must reproduce
                the banked truth row's seeds 16..39 character for character.

    run_mat.sh pin 1 6 45m <abs>/play_e10.py --mode bot|self [--limit N] [--smoke]

Rows: ``<archive>/rows/<mode>/rows.jsonl`` (``--smoke``: ``<archive>/smoke/<mode>/``, snapshot-stratum states played
against ``heuristic``: ENGINEERING only, never read). One fsync'd row per state, resumable, in the registered order
(``order_key``); ``--limit`` caps NEW work. Nothing about an outcome is printed (progress counts only).
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAT = HERE.parent / "hidden_info_maturity_x26_2026-10-18"
sys.path.insert(0, str(MAT))
import mat_common as E  # noqa: E402

E.add_paths()

ANCHOR = "x26_225M"
SALT = "search_e1_0_v1:"
ARCH = Path(os.environ.get("E10_ARCHIVE") or (Path.home() / "gen3ai_archive" / "search_e1_0"))
PLAYOUT = HERE.parents[3] / "src" / "main" / "value_search" / "playout.py"
REF_SEEDS = (16, 40)
SMOKE_BOT = "heuristic"


def order_key(did: str) -> str:
    return hashlib.sha256((SALT + did).encode()).hexdigest()


def bot_seed(did: str) -> int:
    """The bot's route seed: 52 bits of sha256(SALT + "bot:" + id) (exact in a JSON number)."""
    return int(hashlib.sha256((SALT + "bot:" + did).encode()).hexdigest()[:13], 16)


def bot_name(meta: dict) -> str:
    return meta["cycle_opp"].split("/")[1]


def state_ids(smoke: bool):
    meta = E.bank_states()
    want = "snapshot" if smoke else "bot"
    ids = [i for i in E.mode_ids("lean") if meta[i]["stratum"] == want]
    return sorted(ids, key=order_key), meta


def load_playout():
    spec = importlib.util.spec_from_file_location("e10_playout", PLAYOUT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod           # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(mod)
    return mod, hashlib.sha256(PLAYOUT.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("bot", "self"), required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    E.set_bank("interim")
    E.check_pin()
    import common as C  # the pilot's
    import run_playouts as RP  # the pilot's forward (Fwd) and seed plan

    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.reader import inference_globals
    from main.policy_spectrum.truth import cmd_index, to_log, turn_seeds
    from utils.rust_env import ffi as F

    PO, po_sha = load_playout()
    ids, meta = state_ids(a.smoke)
    if a.smoke:
        ids = ids[:a.limit or 4]
    truth = E.read_dir(E.rows_dir(ANCHOR, "O"), "shard_*.jsonl")
    bank = load_bank(E.bank_path())
    dec = {d["id"]: d for d in bank.decisions}
    out_p = ARCH / ("smoke" if a.smoke else "rows") / a.mode / "rows.jsonl"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    done = set(E.read_dir(out_p.parent, "rows.jsonl"))
    todo = [i for i in ids if i not in done and (truth.get(i) or {}).get("ok")]
    if a.limit and not a.smoke:
        todo = todo[:a.limit]
    print(f"[play_e10] {a.mode}{' SMOKE' if a.smoke else ''}: {len(done)} done, {len(todo)} to go "
          f"(of {len(ids)}; {sum(1 for i in ids if not (truth.get(i) or {}).get('ok'))} without an ok truth row)",
          flush=True)
    if not todo:
        return
    lib = F.load(F.default_path("release"))
    drv = os.environ.get("E10_DRIVER_SHA", "")
    t0 = time.monotonic()
    with inference_globals(E.threads()):
        fwd = RP.Fwd(E.load_model(ANCHOR))
        with open(out_p, "a") as f:
            for n, did in enumerate(todo):
                d, r1 = dec[did], truth[did]
                b = bank.battles[bank.battle_index[d["battle"]]]
                side = d["side"]
                at = cmd_index(b, side, d["n"])
                if at != int(r1["at"]) or b.commands[at][1] != d["played"]:
                    raise SystemExit(f"{did}: the root command differs from the banked truth row's")
                seeds = turn_seeds(did, C.CRIT_SEEDS[1])[REF_SEEDS[0]:REF_SEEDS[1]]
                acts = sorted(int(x) for x in r1["out"])
                row = {"id": did, "mode": a.mode, "cp": ANCHOR, "pin": E.series().pin, "driver": drv,
                       "playout_sha256": po_sha, "seeds": list(REF_SEEDS)}
                ob = None
                if a.mode == "bot":
                    name = SMOKE_BOT if a.smoke else bot_name(meta[did])
                    ob = PO.OppBot(name, bot_seed(did))
                    row.update({"bot": name, "bot_seed": ob.seed})
                c0, w0 = C.cpu_seconds(), time.monotonic()
                try:
                    res = PO.referee(fwd, to_log(b), at, side, seeds, actions=acts, opp_bot=ob, lib=lib)
                except Exception as exc:          # a bot refusal or a core error: a REFUSED row (a VOID guard counts it)
                    row.update({"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:400]}"})
                else:
                    tok = {str(k): v for k, v in sorted(res.tokens.items())}
                    row.update({"ok": True, "out": {str(k): v for k, v in sorted(res.out.items())},
                                "tokens_ok": tok == d["tokens"], "other_open": res.other_open,
                                "fwd_rows": res.fwd_rows})
                row["cost"] = {"cpu_s": round(C.cpu_seconds() - c0, 3), "wall_s": round(time.monotonic() - w0, 3),
                               "playouts": len(acts) * len(seeds)}
                E.write_row(f, row)
                el = time.monotonic() - t0
                print(f"[play_e10] {n + 1}/{len(todo)} {el / 60:.1f} min, eta "
                      f"{el / (n + 1) * (len(todo) - n - 1) / 60:.1f} min{'' if row['ok'] else ' REFUSED'}", flush=True)


if __name__ == "__main__":
    main()
