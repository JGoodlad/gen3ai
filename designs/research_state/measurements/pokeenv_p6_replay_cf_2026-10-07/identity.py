"""P6 identity read: the prober's ``replay-counterfactual`` on the poke-env road vs the Rust play-out.

``gen3_pokeenv_p6_replay_cf_identity_v1``. Both roads live in ONE commit, ``6168924c`` (``main/prober/replay.py``:
the new ``replay_counterfactual_battle`` and the LEGACY ``poke_env_replay_counterfactual_battle`` + its
player builders), so every comparison runs the same code base, the same models and the same records.
🚨 **Run it AT ``6168924c``** — the cut-over commit after it deleted the legacy road, so ``run`` / ``dist``
fail at HEAD (``table`` still works on the banked rows).

    # 0. the P5 identity set, copied read-only out of the archive (P5's own builder)
    python ../pokeenv_p5_prober_2026-10-07/capture.py fixture --dir /tmp/p6/fixture
    # 1. two CURRENT-architecture checkpoints (seeded perturbed fresh production policies, CPU, 1 thread)
    python identity.py models --dir /tmp/p6/models
    # 2. the deterministic read: every regime, per ROLLOUT, old vs new (outcome, turns, OUR side's protocol)
    PYTHONHASHSEED=0 python identity.py run --fixture /tmp/p6/fixture --models /tmp/p6/models --out /tmp/p6/run.jsonl
    # 3. the sampled regimes' DISTRIBUTIONS (old: unseeded, the process-wide streams; new: seeded) + rerun
    PYTHONHASHSEED=0 python identity.py dist --fixture /tmp/p6/fixture --models /tmp/p6/models --out /tmp/p6/dist.json
    # 4. the table (the banked rows: rows.jsonl.gz, rows_trained.jsonl.gz)
    python identity.py table rows.jsonl.gz

THE REGIMES. Our side is always the trainee's GREEDY policy. The opponent:

* ``ckpt_greedy``  — a second checkpoint, greedy.
* ``self_greedy``  — the trainee's own checkpoint as the opponent, greedy.
* ``bot``          — the record's opponent when it is a roster bot, else a roster bot by rotation; the
  Python bot's ``_choice_rng`` / ``_protect_rng`` seeded exactly as the core seeds rollout ``k``'s
  bot (``bot_stream_seed(bot_seed, k, s)`` — the Rust bot gate's rule).
* ``ckpt_sampled`` — the second checkpoint at temp 1.0, rollout ``k``'s sampler seeded on BOTH roads
  (the RLPlayer's ``policy_seed`` = the core road's per-rollout ``torch.Generator`` seed).

each at ``n = 1`` (the battle's own dice) and ``n = 4`` (four post-divergence reseeds). A ROLLOUT is
IDENTICAL when its outcome, its last turn and OUR side's protocol lines (every ``|``-line but
``|request|`` — the bridge's JSON request frame, which the core road does not render) are equal.

Pinned (README F-P5-6): ``PYTHONHASHSEED=0`` and ``torch.set_num_threads(4)``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOP = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=HERE, capture_output=True,
                          text=True, check=True).stdout.strip())
sys.path.insert(0, str(TOP / "src"))
sys.path.insert(0, str(HERE.parent / "pokeenv_p5_prober_2026-10-07"))

ROSTER_ROTATION = ("heuristic", "staller", "random", "aggressive_v2", "setup_sweep", "staller_v2",
                   "heuristic2", "aggressive", "setup_sweep_v2")
REGIMES = ("ckpt_greedy", "self_greedy", "bot", "ckpt_sampled")
NS = (1, 4)


def _threads(n: int) -> None:
    import torch

    torch.set_num_threads(n)


def build_models(dst: Path) -> None:
    from agents.training.rust_eval.offline import build_models as bm  # re-homed from the deleted `parity` (P6 slice 6c)

    trainee, sentinels, cfg = bm(dst, n_sentinels=1)
    (dst / "models.json").write_text(json.dumps({"trainee": trainee, "opponent": sentinels[0], "config": cfg}))
    print(json.dumps({"trainee": trainee, "opponent": sentinels[0]}))


def battles(fx: Path):
    """``(run, prefix, summary path)`` for every battle of the P5 identity set."""
    from capture import BATTLES

    for run, prefixes in BATTLES.items():
        for pre in prefixes:
            yield run, pre, str(fx / run / "eval_traces" / (pre + "_summary.json"))


def anchors(summary: dict, npz: dict, k: int = 2):
    """Up to ``k`` move_selection decisions (an early and a late one) with >= 2 legal actions, each with
    a substitute: the first legal action that is NOT the one played."""
    invs = summary["invocations"]
    acts = npz["actions"]
    picks = [i for i, inv in enumerate(invs) if inv.get("phase") == "move_selection" and inv.get("turn", 0) >= 2
             and sum(bool(a.get("valid")) for a in inv["actions"].values()) >= 2]
    if not picks:
        return []
    chosen = sorted({picks[0], picks[len(picks) // 2], picks[-1]})[:k] if k > 1 else [picks[0]]
    out = []
    for i in chosen:
        valid = [j for j, a in enumerate(invs[i]["actions"].values()) if a.get("valid")]
        sub = next((j for j in valid if j != int(acts[i])), valid[0])
        out.append((i, sub))
    return out


def _rename(line: str, names: dict) -> str:
    """Replace each poke-env account name (``CfT3x1`` …, a whole word) by the recorded player name."""
    import re

    pat = re.compile(r"\b(" + "|".join(re.escape(k) for k in names) + r")\b")
    return pat.sub(lambda m: names[m.group(1)], line)


def _proto(lines):
    return [ln for ln in lines if ln.startswith("|") and not ln.startswith("|request|")]


class Roads:
    """The two roads over one session's models and records."""

    def __init__(self, models: dict):
        from main.prober.session import ProbeSession

        self.models = models
        self._sessions = {}
        self._mappings = None
        self.PS = ProbeSession

    def session(self, summary_path: str):
        s = self._sessions.get(summary_path)
        if s is None:
            s = self._sessions[summary_path] = self.PS(summary_path, ckpt_override=self.models["trainee"])
        return s

    def mappings(self):
        if self._mappings is None:
            from agents.observation.state_encoder import load_mappings

            self._mappings = load_mappings()
        return self._mappings

    def case(self, summary_path: str, inv: int, sub: int, regime: str, n: int) -> dict:
        from agents.training.rust_env_opponents import bot_stream_seed
        from main.prober import replay as RP
        from main.prober.core_walk import decision_choices
        from main.prober.falsifier import fresh_seeds
        from utils.rust_env import counterfactual as CF

        s = self.session(summary_path)
        b, record = s._cf_record(summary_path)
        summary = s._summary(b)
        turn = int(summary["invocations"][inv]["turn"])
        our = record.side_of(record.trainee_username)
        opp = {"p1": "p2", "p2": "p1"}[our]
        trainee = s._play_model(self.models["trainee"])
        other = s._play_model(self.models["opponent"])
        tag = record.battle_tag or ""
        post = [None] if n == 1 else list(fresh_seeds(n, salt=f"{tag}:{inv}:cf"))
        sseeds = RP.sampler_seeds(tag, inv, n)
        bseed = RP.bot_seed(tag, inv)
        bot_name = None
        if regime == "bot":
            names = RP._bot_names()
            bot_name = b.opponent if b.opponent in names else ROSTER_ROTATION[int(hashlib.sha256(
                f"{tag}:{inv}".encode()).hexdigest(), 16) % len(ROSTER_ROTATION)]
        opp_model = {"ckpt_greedy": other, "self_greedy": trainee, "ckpt_sampled": other}.get(regime)
        stochastic = regime == "ckpt_sampled"
        choices, _ = decision_choices(record, our, inv)
        sub_choice = choices[sub]

        # --- NEW: the core play-out
        t0 = time.monotonic()
        stall = [our] + ([opp] if opp_model is not None else [])
        res = CF.replay_counterfactual(
            record, divergence_turn=turn, substitute_action=sub, our_policy=CF.ModelPolicy(trainee),
            opp_policy=None if opp_model is None else CF.ModelPolicy(opp_model, stochastic=stochastic,
                                                                     seeds=sseeds if stochastic else None),
            opp_bot=None if bot_name is None else {"name": bot_name, "seed": bseed},
            post_t_seeds=post, stall_sides=stall, text=True)
        new_wall = time.monotonic() - t0
        new = [{"outcome": r.outcome, "turns": r.turns, "proto": _proto(r.text)} for r in res["rollouts"]]

        # --- OLD: the poke-env road, one rollout at a time (the legacy loop), raw chunks captured
        from poke_env.ps_client import LocalhostServerConfiguration
        import utils.bridge.counterfactual as BC

        orig = BC.summarize_trajectory
        BC.summarize_trajectory = lambda side, sink, **kw: [(sd, c) for sd, c in sink]
        old = []
        t0 = time.monotonic()
        try:
            for k, ps in enumerate(post):
                tr = RP.build_trainee(trainee, record, self.mappings(), LocalhostServerConfiguration, tag=f"{inv}x{k}")
                if bot_name is not None:
                    op, _ = RP.build_opponent(bot_name, record, trainee, self.mappings(), LocalhostServerConfiguration,
                                              opponent_source="bot", tag=f"{inv}x{k}",
                                              bot_streams=(bot_stream_seed(bseed, k, 0), bot_stream_seed(bseed, k, 1)))
                else:
                    op, _ = RP.build_opponent("sentinel_x", record, trainee, self.mappings(), LocalhostServerConfiguration,
                                              opp_model=opp_model, opponent_source="ckpt", opponent_ckpt="x",
                                              opponent_stochastic=stochastic, tag=f"{inv}x{k}",
                                              policy_seed=sseeds[k] if stochastic else None)
                r = BC.replay_counterfactual(record, trainee=tr, opponent=op, divergence_turn=turn,
                                             substitute_choice=sub_choice, post_t_seed=ps, capture_trajectory=True,
                                             impl="rust")
                # The poke-env road's players log in as `CfT…` / `CfO…` / `CfB…`; the core plays the
                # RECORDED names (README F-P6-2) — the one textual difference, normalized here.
                names = {tr.username: record.username(our), op.username: record.username(opp)}
                lines = [_rename(ln, names) for sd, c in r["trajectory"] if sd == our for ln in c.split("\n")]
                old.append({"outcome": r["outcome"], "turns": r["turns"], "proto": _proto(lines),
                            "exhausted": r.get("script_exhausted")})
        finally:
            BC.summarize_trajectory = orig
        old_wall = time.monotonic() - t0
        per = []
        for k, (o, w) in enumerate(zip(old, new)):
            first = next((i for i, (a, c) in enumerate(zip(o["proto"], w["proto"])) if a != c),
                         None if len(o["proto"]) == len(w["proto"]) else min(len(o["proto"]), len(w["proto"])))
            per.append({"k": k, "old": o["outcome"], "new": w["outcome"], "old_turns": o["turns"],
                        "new_turns": w["turns"], "proto_equal": first is None, "first_diff": first,
                        "old_line": None if first is None or first >= len(o["proto"]) else o["proto"][first],
                        "new_line": None if first is None or first >= len(w["proto"]) else w["proto"][first],
                        "n_lines": len(w["proto"]), "exhausted": o["exhausted"]})
        identical = all(p["old"] == p["new"] and p["old_turns"] == p["new_turns"] and p["proto_equal"] for p in per)
        return {"battle": summary_path, "inv": inv, "turn": turn, "sub": sub, "sub_choice": sub_choice,
                "regime": regime, "bot": bot_name, "n": n, "our_side": our, "identical": identical,
                "rollouts": per, "new_wall_s": round(new_wall, 3), "old_wall_s": round(old_wall, 3),
                "opp_recorded": res["root"].get("other_recorded")}


def run(args) -> None:
    import numpy as np

    _threads(args.threads)
    models = json.loads((Path(args.models) / "models.json").read_text())
    roads = Roads(models)
    def done_keys():
        keys = set()
        for f in [args.out, *args.skip_from]:
            if Path(f).exists():
                for ln in Path(f).read_text().splitlines():
                    if ln.strip():
                        d = json.loads(ln)
                        keys.add((d["battle"], d["inv"], d["regime"], d["n"]))
        return keys

    out = open(args.out, "a")
    todo = list(battles(Path(args.fixture)))
    if args.reverse:
        todo.reverse()
    for _run, pre, sp in todo:
        if args.only and args.only not in sp:
            continue
        s = roads.session(sp)
        b = s._battle(sp)
        try:
            summary, npz = s._summary(b), s._npz(b)
        except Exception as e:  # noqa: BLE001 — a battle the reader refuses is reported, never skipped silently
            out.write(json.dumps({"battle": sp, "inv": None, "regime": None, "n": None, "error": f"read: {e}"}) + "\n")
            out.flush()
            continue
        if "actions" not in npz:
            out.write(json.dumps({"battle": sp, "inv": None, "regime": None, "n": None, "error": "no actions"}) + "\n")
            continue
        cases = [(inv, sub, regime, n) for inv, sub in anchors(summary, {"actions": np.asarray(npz["actions"])},
                                                                 k=args.per_battle)
                 for regime in REGIMES for n in NS]
        if args.reverse:
            cases.reverse()
        for inv, sub, regime, n in cases:
            if (sp, inv, regime, n) in done_keys():   # re-read: a sibling job may have done it
                continue
            try:
                row = roads.case(sp, inv, sub, regime, n)
            except Exception as e:  # noqa: BLE001
                import traceback

                row = {"battle": sp, "inv": inv, "sub": sub, "regime": regime, "n": n,
                       "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-1500:]}
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(f"{pre} inv={inv} {regime} n={n}: "
                  f"{row.get('identical', row.get('error'))}", flush=True)


def table(paths) -> None:
    import statistics
    from collections import Counter, defaultdict

    seen, rows = {}, []
    rerun_same = rerun_diff = 0
    for path in paths:                         # sibling jobs may have both run a case: keep the first
        import gzip

        text = gzip.open(path, "rt").read() if str(path).endswith(".gz") else Path(path).read_text()
        for ln in text.splitlines():
            if ln.strip():
                d = json.loads(ln)
                key = (d["battle"], d["inv"], d["regime"], d["n"])
                sig = [(p["new"], p["new_turns"], p["n_lines"]) for p in d.get("rollouts", [])]
                if key not in seen:
                    seen[key] = sig
                    rows.append(d)
                elif seen[key] == sig:         # the NEW road run twice, in two processes: reproducible?
                    rerun_same += 1
                else:
                    rerun_diff += 1
                    print("RERUN DIFFERS", key)
    ok = [r for r in rows if "error" not in r]
    ro = [p for r in ok for p in r["rollouts"]]
    print(f"battles {len({r['battle'] for r in ok})}, decisions {len({(r['battle'], r['inv']) for r in ok})}, "
          f"rollouts {len(ro)}, protocol lines compared {sum(p['n_lines'] for p in ro)}")
    print("outcomes (new):", dict(Counter(p["new"] for p in ro)), "; rollouts >= 250 turns:",
          sum(p["new_turns"] >= 250 for p in ro), "; turns median / max:", statistics.median(p["new_turns"] for p in ro),
          max(p["new_turns"] for p in ro))
    print("our side:", dict(Counter(r["our_side"] for r in ok)), "; bots:",
          dict(Counter(r["bot"] for r in ok if r["regime"] == "bot")))
    print("old road script exhausted:", dict(Counter(tuple(p["exhausted"] or ()) for p in ro)))
    nw, ow = sum(r["new_wall_s"] for r in ok), sum(r["old_wall_s"] for r in ok)
    print(f"wall: new {nw:.0f} s, old {ow:.0f} s, new/old {nw / max(ow, 1e-9):.3f}")
    print(f"cases the new road ran TWICE (sibling jobs, separate processes): {rerun_same} identical, {rerun_diff} differ")
    print()

    agg = defaultdict(lambda: [0, 0, 0, 0])   # cases, identical, rollouts, rollouts identical
    errs = [r for r in rows if "error" in r]
    for r in rows:
        if "error" in r:
            continue
        a = agg[(r["regime"], r["n"])]
        a[0] += 1
        a[1] += int(r["identical"])
        a[2] += len(r["rollouts"])
        a[3] += sum(int(p["old"] == p["new"] and p["old_turns"] == p["new_turns"] and p["proto_equal"])
                    for p in r["rollouts"])
    print("| regime | n | cases | cases identical | rollouts | rollouts identical |")
    print("|---|---|---|---|---|---|")
    for (reg, n), a in sorted(agg.items()):
        print(f"| `{reg}` | {n} | {a[0]} | {a[1]} | {a[2]} | {a[3]} |")
    print(f"\nerrors: {len(errs)}")
    for e in errs:
        print(" -", e["battle"].split("eval_traces/")[-1], e.get("inv"), e.get("regime"), e.get("n"), e["error"][:300])
    for r in rows:
        if "error" not in r and not r["identical"]:
            print("DIFF", r["battle"].split("eval_traces/")[-1], r["inv"], r["regime"], r["n"],
                  [(p["k"], p["old"], p["new"], p["old_turns"], p["new_turns"], p["first_diff"], p["old_line"],
                    p["new_line"]) for p in r["rollouts"] if not (p["old"] == p["new"] and p["proto_equal"])][:2])


def dist(args) -> None:
    """The SAMPLED regime's distribution: old (unseeded RLPlayer, the process-wide torch stream) vs new
    (seeded), on a few decisions at ``n`` rollouts each; and the new road rerun (reproducibility)."""
    import numpy as np

    from main.prober import replay as RP
    from main.prober.replay import wilson_ci

    _threads(args.threads)
    models = json.loads((Path(args.models) / "models.json").read_text())
    roads = Roads(models)
    from poke_env.ps_client import LocalhostServerConfiguration

    cases = []
    for _run, pre, sp in battles(Path(args.fixture)):
        if not any(k in pre for k in args.battles.split(",")):
            continue
        s = roads.session(sp)
        b = s._battle(sp)
        summary, npz = s._summary(b), s._npz(b)
        for inv, sub in anchors(summary, {"actions": np.asarray(npz["actions"])}, k=1):
            _b, record = s._cf_record(sp)
            trainee = s._play_model(models["trainee"])
            other = s._play_model(models["opponent"])
            kw = dict(play_model=trainee, opp_name="sentinel_x", opponent_ckpt="x", opp_model=other,
                      opponent_source="ckpt", opponent_stochastic=True, n_rollouts=args.n)
            new1 = RP.replay_counterfactual_battle(record, summary, npz, inv, sub, **kw)
            new2 = RP.replay_counterfactual_battle(record, summary, npz, inv, sub, **kw)
            old = RP.poke_env_replay_counterfactual_battle(record, summary, npz, inv, sub, mappings=roads.mappings(),
                                                           server_config=LocalhostServerConfiguration, impl="rust", **kw)
            n = args.n
            p_new, p_old = new1["wins"] / n, old["wins"] / n
            se = ((p_new * (1 - p_new) + p_old * (1 - p_old)) / n) ** 0.5
            cases.append({"battle": sp, "inv": inv, "sub": sub, "n": n,
                          "new": {"wins": new1["wins"], "outcomes": new1["outcomes"], "ci": list(wilson_ci(new1["wins"], n))},
                          "new_rerun_identical": {k: v for k, v in new1.items() if k != "play_out"}
                          == {k: v for k, v in new2.items() if k != "play_out"},
                          "old_unseeded": {"wins": old["wins"], "outcomes": old["outcomes"], "ci": list(wilson_ci(old["wins"], n))},
                          "diff": round(p_new - p_old, 4), "diff_ci95": [round(p_new - p_old - 1.96 * se, 4),
                                                                          round(p_new - p_old + 1.96 * se, 4)]})
            print(json.dumps(cases[-1]), flush=True)
    Path(args.out).write_text(json.dumps(cases, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("models")
    m.add_argument("--dir", required=True)
    r = sub.add_parser("run")
    r.add_argument("--fixture", required=True)
    r.add_argument("--models", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--per-battle", type=int, default=2)
    r.add_argument("--threads", type=int, default=4)
    r.add_argument("--only", default="")
    r.add_argument("--reverse", action="store_true", help="walk the battles and cases from the END (a sibling job)")
    r.add_argument("--skip-from", nargs="*", default=[], help="other jobs' row files: their cases are skipped")
    d = sub.add_parser("dist")
    d.add_argument("--fixture", required=True)
    d.add_argument("--models", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--n", type=int, default=60)
    d.add_argument("--battles", default="loss_s0_005,win_s2_002,loss_s0_003")
    d.add_argument("--threads", type=int, default=4)
    t = sub.add_parser("table")
    t.add_argument("paths", nargs="+")
    a = ap.parse_args()
    if a.cmd == "models":
        build_models(Path(a.dir))
    elif a.cmd == "run":
        run(a)
    elif a.cmd == "dist":
        dist(a)
    else:
        table(a.paths)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    raise SystemExit(main())
