"""STAGE 1, PLAY SIDE: every arm's worlds PLAYED with X26's 225M checkpoint at the PIN checkout (``run_s1.sh pin``: X26's
pin + the damage-roll control, ``s1_common`` docstring), by the value-discrimination pilot's own Phase 2 driver
(``run_det_playouts.one_state``, imported unchanged, as the maturity battery plays its arms): per state every legal
action played to the END in each of the 8 worlds (world j on dice seed j), the checkpoint GREEDY on both sides, the
searched side's value trace, and the honest one-ply critic (world j, critic seed 40 + j).

THE ROLLS (F1 tier A; the brief's known hazard: a world rebuilt from its team alone loses its rolls): the pilot's
``world_log`` rebuilds each world's core log from the bank battle, the world's team and its root prefix. For a world
built with ``dice="rolls"`` the force table its gate accepted is added to that log as ``"rolls"`` (the playout tool
replays it), through a wrapper that checks it is called once per world, in world order, with that world's team. A world
replayed without its rolls would reach the root on another history: ``one_state`` REFUSES any world whose root tokens
differ from the banked root's (a refusal row, never a silent mismatch).

    run_s1.sh pin 1 8 6h play_s1.py --ids <id,...>            # the arms' rows
    run_s1.sh pin 1 8 2h play_s1.py --identity                # THE IDENTITY TOOTH (below)

THE IDENTITY TOOTH (``--identity``): 12 fixed states of the 287 (sha256 order, salt ``ws_stage1_identity_v1:``):
their BANKED F worlds (the maturity battery's, built at 826bee89) re-played here must reproduce the BANKED 225M F rows
(played at the pin) exactly: ``out``, ``trace``, ``crit``, ``other_open``. It proves this checkout plays unarmed worlds
exactly as the pin did. Only counts are printed.

One fsync'd row per (state, arm) in ``<archive>/rows/<arm>/rows.jsonl``; resumable. A world row that is an exclusion
or a defect gets an ``ok: false`` row carrying the builder's reason. No outcome is printed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s1_common as E  # noqa: E402

for _p in (E.T1, E.BRC, E.PILOT):
    if str(_p) not in sys.path:
        sys.path.insert(1, str(_p))

N_IDENTITY = 12
PIN_ROLLS_SHA_FILE = E.CACHE / "pin_rolls.sha"


def check_pin() -> str:
    if os.environ.get("WS1_KIND") != "pin":
        raise SystemExit("worlds are played at the PIN checkout only (run_s1.sh pin)")
    head = os.environ["WS1_CHECKOUT_HEAD"]
    import subprocess

    d = os.environ["WS1_CHECKOUT_DIR"]

    def git(*args) -> str:
        return subprocess.run(["git", "-C", d, *args], capture_output=True, text=True, check=True).stdout.strip()

    if git("rev-parse", "HEAD~1") != E.X26_PIN:
        raise SystemExit(f"the play checkout's parent is not X26's pin {E.X26_PIN}")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("the play checkout has local changes")
    # HEAD = the pin + ONE commit, and that commit's diff IS the roll-control commit's diff on the Rust sources
    for path in ("src/rust_sim/src", "src/rust_env/src"):
        if git("diff", "HEAD~1", "HEAD", "--", path) != git("diff", f"{E.ROLL_CONTROL_COMMIT}~1", E.ROLL_CONTROL_COMMIT,
                                                            "--", path):
            raise SystemExit(f"the play checkout's {path} change is not the damage-roll control's ({E.ROLL_CONTROL_COMMIT})")
    import main

    if not str(Path(main.__file__).resolve()).startswith(d):
        raise SystemExit(f"main imported from {main.__file__}, not the pin checkout {d}")
    return head


def identity_ids():
    return sorted(E.common_ids(), key=lambda x: hashlib.sha256(("ws_stage1_identity_v1:" + x).encode()).hexdigest()
                  )[:N_IDENTITY]


class RollsLog:
    """``run_det_playouts.world_log`` with the world's rolls added (module docstring): set :attr:`row` before each
    ``one_state`` call; the wrapper checks it is called once per world, in order, with that world's team."""

    def __init__(self, orig):
        self.orig = orig
        self.row = None
        self.calls = 0

    def __call__(self, b, team_w, cmds_w, opp_key):
        lg = self.orig(b, team_w, cmds_w, opp_key)
        r = self.row
        if r is None:
            raise RuntimeError("world_log called with no world row set")
        j = self.calls
        if j >= E.J or team_w != r["teams"][j] or cmds_w != r["cmds"][j]:
            raise RuntimeError(f"world_log call {j} does not match the row's world {j}")
        self.calls += 1
        rolls = (r.get("rolls") or [[]] * E.J)[j]
        if rolls:
            lg["rolls"] = [list(x) for x in rolls]
        return lg

    def arm(self, row) -> None:
        self.row, self.calls = row, 0

    def check_done(self) -> None:
        if self.row is not None and self.calls != E.J:
            raise RuntimeError(f"world_log called {self.calls} times for {E.J} worlds")
        self.row = None


def load_model():
    from main.policy_spectrum.qhat import check_winprob
    from main.policy_spectrum.reader import load_checkpoint

    e = E.anchor_ckpt()
    p = Path(e["file"])
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    if h != e["sha256"]:
        raise SystemExit(f"{p}: sha256 {h} differs from the registered {e['sha256']}")
    m = load_checkpoint(p, device="cpu")
    check_winprob(m)
    return m, e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default=None)
    ap.add_argument("--identity", action="store_true")
    ap.add_argument("--arms", default=",".join(E.ARMS))
    ap.add_argument("--smoke", action="store_true", help="the engineering smoke's states (a *_smoke archive only)")
    a = ap.parse_args()
    head = check_pin()
    import common as C  # the pilot's: clock offset
    import run_det_playouts as RDP
    from run_playouts import Fwd

    if RDP.J != E.J:
        raise SystemExit(f"the pilot driver plays {RDP.J} worlds, stage 1 registers {E.J}")
    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.reader import inference_globals
    from main.policy_spectrum.truth import cmd_index
    from utils.rust_env import ffi as F
    from utils.rust_env.successors import SearchCore, SuccessorsError

    bank = load_bank(E.MAT_ARCH / "bank")
    dec = {d["id"]: d for d in bank.decisions}
    if a.identity:
        ids = identity_ids()
        jobs = [("F", did) for did in ids]
        worlds = {"F": E.read_by_id(E.MAT_ARCH / "worlds" / "shared" / "F" / "worlds_0_of_1.jsonl", first=True)}
        out_paths = {"F": E.d("identity") / "rows.jsonl"}
        done = {"F": E.read_by_id(out_paths["F"])}
    else:
        if a.smoke and not str(E.ARCH).endswith("_smoke"):
            raise SystemExit("the engineering smoke writes only to a *_smoke archive (WS1_ARCHIVE)")
        ids = [x for x in (a.ids or "").split(",") if x]
        if not a.smoke:
            allowed = set(E.common_ids())
            if any(x not in allowed for x in ids):
                raise SystemExit("an id outside the 287 registered states")
        arms = [x for x in a.arms.split(",") if x]
        worlds = {arm: E.read_by_id(E.worlds_path(arm)) for arm in arms}
        out_paths = {arm: E.rows_path(arm) for arm in arms}
        done = {arm: E.read_by_id(out_paths[arm]) for arm in arms}
        jobs = [(arm, did) for did in ids for arm in arms if did in worlds[arm] and did not in done[arm]]
    print(f"[s1 play] {len(jobs)} (arm, state) jobs at {head[:8]}", flush=True)
    if not jobs:
        return
    off = C.clock_offset()
    lib = F.load(F.default_path("release"))
    cores = [SearchCore(lib=lib) for _ in range(E.J)]
    rl = RollsLog(RDP.world_log)
    RDP.world_log = rl
    files = {arm: open(p, "a") for arm, p in out_paths.items()}
    t0 = time.monotonic()
    try:
        with inference_globals(1):
            model, ck = load_model()
            fwd = Fwd(model)
            for n_i, (arm, did) in enumerate(jobs):
                w = worlds[arm][did]
                d = dec[did]
                if not w.get("ok"):
                    r = {"id": did, "ok": False, "excluded": not w.get("crash"), "defect": bool(w.get("crash")),
                         "error": "world build: " + str(w.get("error"))[:500]}
                elif len(w["teams"]) != E.J or any(t is None for t in w["teams"]):
                    r = {"id": did, "ok": False, "error": f"world row holds {len(w['teams'])} worlds, not {E.J}"}
                elif cmd_index(bank.battles[bank.battle_index[d["battle"]]], w["side"], d["n"]) != int(w["at"]):
                    r = {"id": did, "ok": False, "error": f"the pin's root command index != the builder's {w['at']}"}
                else:
                    rl.arm(w)
                    try:
                        r = RDP.one_state(cores, fwd, bank, did, w, off)
                        rl.check_done()
                    except SuccessorsError as exc:
                        rl.row = None
                        r = {"id": did, "ok": False, "error": str(exc)[:500]}
                        for k, core in enumerate(cores):       # a refused open leaves no state; reopen all
                            core.close()
                            cores[k] = SearchCore(lib=lib)
                r.update({"arm": arm, "cp": E.ANCHOR, "step": ck["step"], "pin_head": head,
                          "world_head": w.get("head"), "log_w": w.get("log_w") or []})
                E.write_row(files[arm], r)
                if not r.get("ok"):
                    print(f"[s1 play] {arm} REFUSED {did}: {str(r.get('error'))[:200]}", flush=True)
                el = time.monotonic() - t0
                print(f"[s1 play] {n_i + 1}/{len(jobs)} {el / 60:.1f} min", flush=True)
    finally:
        for f in files.values():
            f.close()
        for core in cores:
            core.close()
    if a.identity:
        banked = E.read_by_id(E.MAT_ARCH / "rows" / E.ANCHOR / "F" / "rows_0_of_1.jsonl")
        got = E.read_by_id(out_paths["F"])
        same = sum(1 for did in ids if did in got and did in banked
                   and all(json.dumps(got[did].get(k), sort_keys=True) == json.dumps(banked[did].get(k), sort_keys=True)
                           for k in ("out", "trace", "crit", "other_open")))
        res = {"states": len(ids), "identical": same, "ids_sha256": hashlib.sha256("".join(ids).encode()).hexdigest(),
               "play_head": head}
        (E.d("identity") / "identity.json").write_text(json.dumps(res, indent=1) + "\n")
        print(f"[s1 identity] {same} / {len(ids)} identical", flush=True)
        if same != len(ids):
            raise SystemExit("THE IDENTITY TOOTH FAILED: this checkout does not play unarmed worlds as the pin did")


if __name__ == "__main__":
    main()
