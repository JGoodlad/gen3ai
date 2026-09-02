"""M9b — WHAT DID v8's LAST ~2.5M FOLD STEPS UNDO? The peak->final behavioural fingerprint.

QUESTION. `v8_gift_timing_2026-09-01.md` measured that v8's fold (`ai_v8_14_distill3_0725`)
GIFTED untaught pool teams a gain that PEAKS at **+9.67pp** at `checkpoint_290115536`
(+12.53M fold steps) and then DECAYS monotonically to **+4.98pp** at `final_model_interrupted`
(+15.04M) — 9.67 -> 8.25 -> 7.03 -> 4.98 over four consecutive arms. The fold ran
`--distill-coef 1.0` against the same three teachers throughout. WHAT did the last 2.5M steps
UNDO? That question is about BEHAVIOUR, and behaviour is the one trace that ports across the
ai_v9 architecture rewrite (ledger d392e80).

INSTRUMENT — REUSED, not rebuilt. This is `v8_fold_behavioral_fingerprint_probe.py` (M4) with
THREE arms instead of two. Every cell-defining constant is IMPORTED from that file (which copied
them from probe P's `/tmp/probeP/selection.json`): the 16 untaught probe teams, the 6 taught
controls, the 8 fixed opponent teams, the fixed reference opponent `ai_v8_03_zarch_control_0718`
(an ancestor of every arm, equal to none), the per-cell CRN seed construction
`random.Random(f"{team}:{opp}")`, greedy play both sides, the node bridge, CPU. Game index i is
therefore the SAME battle for every arm, and these battles are a CRN prefix subsample of probe
P's own — the same subsample M4 used.

THE THREE ARMS
  parent  ai_v8_04_distill_4teacher_0722/final_model_interrupted.zip   277,583,267  (fork source)
  peak    ai_v8_14_distill3_0725/checkpoints/checkpoint_290115536_...   290,115,536  (+12.53M)
  final   ai_v8_14_distill3_0725/final_model_interrupted.zip            292,623,779  (+15.04M)

EVERY arm ACTS on the full cell grid, and at EVERY decision ALL THREE arms are scored on the
IDENTICAL (obs, action_mask). That gives three delta vectors per team class over the same rows:
parent->peak, peak->final, parent->final. `parent` and `final` are M4's own two arms, so M4's
published untaught/taught vectors are reproduced by this run as a cross-check.

ERA PIN. The v8 arms load only under the v8-era code
(`b13b30b289c5eaba136a930a4ab63451e209fbe5`); run from a PRIVATE copy of the era checkout. The
era's rust bridge predates the seedless-seed fix (`bc00d4d`), so `--impl node` is mandatory.

THE FIVE REPRODUCIBILITY SEEDS DO NOT EXIST AT THIS COMMIT (grep over the era tree finds zero
references; they landed 2026-08-30). Determinism comes from the same three things it came from
for probe P, M4 and the timing probe: `stochastic=False` (no policy draw), one PINNED team per
side (no team draw), an EXPLICIT 4-int sim seed per battle (no dice draw).

Run (from the era-pinned private copy):
  PYTHONPATH=<era>/src GEN3AI_TIMEOUT_SCALE=12 nice -n 15 python <this> \
      --kind untaught --shard 0/3 --out /tmp/m9b/rows
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import itertools
import json
import os
import random
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import torch as th  # noqa: E402

th.set_num_threads(1)

from poke_env.player.battle_order import DefaultBattleOrder, ForfeitBattleOrder  # noqa: E402
from poke_env.ps_client import AccountConfiguration, LocalhostServerConfiguration  # noqa: E402

from agents.inference.player import RLPlayer  # noqa: E402
from agents.model.snapshot import current_model_version, load_foreign_opponent  # noqa: E402
from agents.observation.state_encoder import load_mappings  # noqa: E402
from utils.bridge.local_battle_runner import run_local_battles  # noqa: E402
from utils.team_loader import TeamLoader  # noqa: E402
from utils.teambuilder import Gen3Teambuilder  # noqa: E402

# The M4 classifier and selection are IMPORTED rather than re-typed: an axis basis or a cell set
# that drifted between the two probes would make every cosine below incomparable to M4's
# published vectors, which is the whole point of reusing the instrument.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v8_fold_behavioral_fingerprint_probe import (  # noqa: E402
    V8_SELECTION,
    classify_decision,
)

MAIN = "/home/goodlad/dev/gen3ai"
MD = f"{MAIN}/models"
FOLD_RUN = f"{MD}/ai_v8_14_distill3_0725"
CFG = f"{FOLD_RUN}/model_config.json"

ARMS = {
    "parent": (f"{MD}/ai_v8_04_distill_4teacher_0722/final_model_interrupted.zip",
               f"{MD}/ai_v8_04_distill_4teacher_0722/model_config.json", 277_583_267),
    "peak":   (f"{FOLD_RUN}/checkpoints/checkpoint_290115536_steps.zip", CFG, 290_115_536),
    "final":  (f"{FOLD_RUN}/final_model_interrupted.zip", CFG, 292_623_779),
}
ARM_ORDER = ("parent", "peak", "final")
REF = (f"{MD}/ai_v8_03_zarch_control_0718/final_model_interrupted.zip",
       f"{MD}/ai_v8_03_zarch_control_0718/model_config.json")


class TriScoringPlayer(RLPlayer):
    """Acts with ``model``; scores EVERY arm in ``others`` on the SAME (obs, mask).

    The extra forwards are the whole instrument: they make the comparison an IDENTICAL-BOARD
    comparison rather than a comparison of three different games. M4's two-arm
    ``DualScoringPlayer`` is the direct ancestor; the only change is that the scored set is a
    dict rather than one model.
    """

    def __init__(self, *a, others=None, sink=None, meta=None, **kw):
        super().__init__(*a, **kw)
        self._others = dict(others or {})
        self._sink = sink
        self._meta = dict(meta or {})
        self._pending: dict | None = None
        self.n_rows = 0

    def _predict_best_action(self, battle, stochastic=False, need_aux=True, temperature=1.0):
        # need_aux is FORCED true: `choose_move` calls with need_aux=False (it wants only the
        # index), but the recorder needs the acting arm's own logits/value on the identical obs.
        idx, probs, mask = super()._predict_best_action(
            battle, stochastic=stochastic, need_aux=self._sink is not None,
            temperature=temperature)
        if idx is None or self._sink is None:
            return idx, probs, mask
        try:
            self._pending = self._build_row(battle, idx, mask)
        except Exception as e:  # a recorder fault must never change the battle
            print(f"    [rec-fail] {type(e).__name__}: {e}", flush=True)
            self._pending = None
        return idx, probs, mask

    def choose_move(self, battle):
        # A row is written ONLY for the decision that was actually SENT. `choose_move` may
        # re-decide (the stale-request race) and each attempt runs a forward; without this the
        # superseded attempts would enter the behavioural tables as if they were played.
        if self._sink is None:
            return RLPlayer.choose_move(self, battle)
        self._pending = None
        order = super().choose_move(battle)
        row, self._pending = self._pending, None
        # A DEFAULT order means the decision was NOT the model's; `DefaultBattleOrder.order` is
        # the non-None string `/choose default`, so the test has to be on the TYPE.
        deferred = isinstance(order, (DefaultBattleOrder, ForfeitBattleOrder))
        if row is not None and not deferred:
            self._sink.write((json.dumps(row, separators=(",", ":")) + "\n").encode())
            self.n_rows += 1
        return order

    def _build_row(self, battle, idx: int, mask) -> dict | None:
        snap = getattr(self, "_last_prediction", None)
        if snap is None:
            return None
        obs = snap["obs"]
        m = np.asarray(mask, dtype=np.float32)
        obs_t = th.as_tensor(obs[None, :])
        mask_t = th.as_tensor(m[None, :])
        pin = {"observation": obs_t, "action_mask": mask_t}
        neg = (mask_t - 1.0) * 1e9

        acting = self._meta["arm"]
        act_logits = th.as_tensor(snap["logits"])[None, :] + neg
        idx_by_arm = {acting: int(idx)}
        v_by_arm = {acting: round(float(snap["value"]), 5)}
        p_by_arm = {acting: [round(float(x), 4)
                             for x in th.softmax(act_logits, dim=1)[0].numpy()]}
        with th.no_grad():
            for nm, mdl in self._others.items():
                od = mdl.policy.get_distribution(pin)
                ol = od.distribution.logits + neg
                p_by_arm[nm] = [round(float(x), 4) for x in th.softmax(ol, dim=1)[0].numpy()]
                v_by_arm[nm] = round(float(mdl.policy.predict_values(pin)[0].item()), 5)
                idx_by_arm[nm] = int(np.argmax(np.where(m > 0, ol[0].numpy(), -1e30)))

        ctx = self._get_tracker(battle).last_ctx
        d = classify_decision(ctx.legal, battle.live_view())
        d.update(self._meta)
        d["tag"] = battle.strict_view().battle_tag
        d["act_idx"] = int(idx)
        # The per-arm argmax on the IDENTICAL board — the object every axis reads.
        d["idx"] = {k: idx_by_arm[k] for k in ARM_ORDER}
        d["v"] = {k: v_by_arm[k] for k in ARM_ORDER}
        d["p"] = {k: p_by_arm[k] for k in ARM_ORDER}
        d["our_hp"] = round(d["our_hp"], 4)
        d["opp_hp"] = round(d["opp_hp"], 4)
        return d


_ACCT = itertools.count(1)


def _acct(tag: str) -> AccountConfiguration:
    return AccountConfiguration(f"M9{tag[:2]}{next(_ACCT):05d}", "pw")


def sha10(s: str) -> str:
    return hashlib.sha1(s.strip().encode()).hexdigest()[:10]


def load(zip_path: str, cfg: str, cv):
    m, _ = load_foreign_opponent(zip_path, current_version=cv, device="cpu", config_path=cfg)
    fe = m.policy.features_extractor
    if hasattr(fe, "_debugger"):
        fe._debugger = None
    m.policy.set_training_mode(False)
    return m


def acid_test(models: dict) -> dict:
    """The arms LOAD and are DISTINCT networks. A mis-resolved path that loads one zip twice
    reads as a perfect null, so distinctness is a GATE, not a nicety."""
    sds = {n: m.policy.state_dict() for n, m in models.items()}
    names = list(models)
    keys = sorted(set.intersection(*[set(s) for s in sds.values()]))
    pmat = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            tot = 0.0
            for k in keys:
                ta, tb = sds[a][k], sds[b][k]
                if ta.shape == tb.shape and ta.is_floating_point():
                    tot += float((ta - tb).pow(2).sum())
            pmat[f"{a}|{b}"] = round(tot ** 0.5, 4)
    return {"pairwise_param_l2": pmat, "all_distinct": all(v > 1e-3 for v in pmat.values()),
            "n_params": {n: int(sum(p.numel() for p in m.policy.parameters()))
                         for n, m in models.items()}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="untaught", choices=("untaught", "taught"))
    ap.add_argument("--games", type=int, default=4, help="battles per (team, opp, arm) cell")
    ap.add_argument("--opps", type=int, default=8, help="how many of the 8 fixed opponents")
    ap.add_argument("--impl", default="node", choices=("node", "rust"))
    ap.add_argument("--shard", default="0/1", help="i/k over the probe-team list")
    ap.add_argument("--out", default="/tmp/m9b/rows")
    a = ap.parse_args(argv)
    if a.impl != "node":
        raise SystemExit("[m9b] the v8 era predates the rust seedless-seed fix — node is mandatory")

    t0 = time.time()
    mappings = load_mappings()
    cv = current_model_version(mappings)
    models = {k: load(*ARMS[k][:2], cv) for k in ARM_ORDER}
    models["ref"] = load(*REF, cv)
    acid = acid_test(models)
    print(f"[m9b] ACID {json.dumps(acid)}", flush=True)
    if not acid["all_distinct"]:
        raise SystemExit("[m9b] ACID FAILED — arms are not distinct networks")

    sel = V8_SELECTION
    pool = {sha10(t): t for t in TeamLoader().get_all_teams()}
    key = "probe_untaught" if a.kind == "untaught" else "control_taught"
    missing = [s for s in sel[key] + sel["opponents"] if s not in pool]
    if missing:
        raise SystemExit(f"[m9b] selection GIGO: {missing} not in the 719-team pool")

    probes = list(sel[key])
    si, sk = (int(x) for x in a.shard.split("/"))
    probes = [p for i, p in enumerate(probes) if i % sk == si]
    opps = sel["opponents"][:a.opps]
    print(f"[m9b] kind={a.kind} {len(probes)} teams x {len(opps)} opps x {a.games}g x 3 arms = "
          f"{len(probes) * len(opps) * a.games * 3} battles", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tag = f"{a.kind}_s{si}of{sk}"
    cell_path = f"{a.out}_{tag}_cells.jsonl"
    # RESUME. Every cell is INDEPENDENT (its seeds come from an identity-derived generator and
    # each battle is played on an explicit seed under greedy play), so re-entering with earlier
    # cells present reproduces the uninterrupted run.
    done = set()
    if os.path.exists(cell_path):
        for ln in open(cell_path):
            try:
                d = json.loads(ln)
            except Exception:
                continue
            if d.get("requested") == d.get("finished"):
                done.add((d["team"], d["opp"], d["arm"]))
    sink = gzip.open(f"{a.out}_{tag}.jsonl.gz", "ab")
    cellsink = open(cell_path, "a")

    n_rows = 0
    for team_sha in probes:
        team_str = pool[team_sha]
        for opp_sha in opps:
            # CRN: probe P's own per-cell seed construction, so these battles ARE probe P's.
            rng = random.Random(f"{team_sha}:{opp_sha}")
            seeds = [[rng.randrange(0, 65536) for _ in range(4)] for _ in range(a.games)]
            for arm in ARM_ORDER:
                if (team_sha, opp_sha, arm) in done:
                    print(f"  skip (done) {team_sha} vs {opp_sha} [{arm}]", flush=True)
                    continue
                ts = time.time()
                meta = {"team": team_sha, "kind": a.kind, "opp": opp_sha, "arm": arm,
                        "arch": sel["labels"].get(team_sha)}
                others = {k: models[k] for k in ARM_ORDER if k != arm}
                p = TriScoringPlayer(
                    model=models[arm], team=Gen3Teambuilder([team_str]),
                    battle_format="gen3ou", server_configuration=LocalhostServerConfiguration,
                    mappings=mappings, account_configuration=_acct(arm),
                    stochastic=False, start_listening=False,
                    others=others, sink=sink, meta=meta)
                o = RLPlayer(model=models["ref"], team=Gen3Teambuilder([pool[opp_sha]]),
                             battle_format="gen3ou",
                             server_configuration=LocalhostServerConfiguration,
                             mappings=mappings, account_configuration=_acct("rf"),
                             stochastic=False, start_listening=False)
                # PER-GAME outcomes + the battle tag they belong to. All three arms play the same
                # seed list, so game index i is the SAME battle for all of them.
                per_game, tags, seen = [], [], set()
                for s in seeds:
                    w0, f0 = p.n_won_battles, p.n_finished_battles
                    try:
                        asyncio.run(run_local_battles(p, o, 1, seed=list(s), concurrency=1,
                                                      impl=a.impl))
                    except Exception as e:
                        print(f"    !! {arm}/{team_sha}/{opp_sha} {type(e).__name__}: "
                              f"{str(e)[:120]}", flush=True)
                    new = [t for t in p.battles if t not in seen]
                    seen.update(new)
                    tags.append(new[0] if len(new) == 1 else None)
                    per_game.append(1 if p.n_won_battles > w0
                                    else (0 if p.n_finished_battles > f0 else -1))
                rec = {"team": team_sha, "kind": a.kind, "opp": opp_sha, "arm": arm,
                       "arch": sel["labels"].get(team_sha), "step": ARMS[arm][2],
                       "wins": p.n_won_battles, "finished": p.n_finished_battles,
                       "requested": a.games, "rows": p.n_rows,
                       # The two deferral counters. `n_redecides > 0` is the only path on which
                       # the recorder could see a SUPERSEDED decision, so recording them makes
                       # the write-guard's correctness a MEASUREMENT rather than a claim.
                       "n_defaults": p._n_defaults, "n_redecides": p._n_redecides,
                       "n_decisions": p._n_decisions,
                       "per_game": per_game, "tags": tags,
                       "secs": round(time.time() - ts, 1)}
                cellsink.write(json.dumps(rec) + "\n")
                cellsink.flush()
                sink.flush()
                n_rows += p.n_rows
                print(f"  {a.kind:8s} {team_sha} vs {opp_sha} [{arm:6s}] "
                      f"{p.n_won_battles}/{p.n_finished_battles} rows={p.n_rows} "
                      f"{rec['secs']:.0f}s (elapsed {time.time() - t0:.0f}s)", flush=True)
    sink.close()
    cellsink.close()
    print(f"[m9b] done: {n_rows} decision rows in {time.time() - t0:.0f}s", flush=True)
    with open(f"{a.out}_{tag}_meta.json", "w") as f:
        json.dump({"acid": acid, "kind": a.kind, "games": a.games, "opps": opps,
                   "shard": a.shard, "impl": a.impl, "n_rows": n_rows,
                   "arms": {k: {"path": ARMS[k][0], "step": ARMS[k][2]} for k in ARM_ORDER},
                   "wall_s": round(time.time() - t0, 1)}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
