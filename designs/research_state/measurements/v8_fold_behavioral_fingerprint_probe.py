"""M4 — the BEHAVIORAL FINGERPRINT of v8's gift to UNTAUGHT teams.

Probe P proved the v8_14 fold GIFTED untaught pool teams (+5.42pp, z=4.83, 14/16 teams
positive) while gaining +26.18pp on the teams it taught. It said nothing about WHAT the fold
made the model DO differently. This probe answers that in BEHAVIOUR space — the one trace that
ports across an architecture rewrite, where parameter/representation traces do not (ledger
d392e80).

THE MEASUREMENT — dual scoring on IDENTICAL boards
--------------------------------------------------
Probe P's cells are re-played (same probe teams, same fixed reference opponent, same CRN seeds)
with PER-DECISION LOGGING. At every decision of the acting arm's trajectory the OTHER arm is
scored on the SAME (obs, mask) — one extra forward, no extra battles — so every row carries:

    what the acting arm did  ·  what the other arm WOULD have done  ·  the board it happened on

Both arms act, so both state distributions are covered: A-on-A vs B-on-A (the parent's board
distribution) and B-on-B vs A-on-B (the fold's). A one-sided read would confound the policy
change with the state-distribution shift it causes.

Each decision is classified into ACTION CLASSES from the server-authoritative ``LegalActions``
snapshot plus ``gen3_data`` move facts (never from the obs vector), and into BOARD STRATA
(matchup sign, our HP, phase of game). The per-axis effect is then the PAIRED difference of the
two arms' argmax rates over the same rows, battle-cluster bootstrapped.

TAUGHT vs UNTAUGHT is the deliverable: the same axis vector is measured on both slices and
compared for SHAPE (cosine, origin-through slope, per-axis sign agreement). "Weaker version of
the same fingerprint" and "a different fingerprint" are different geometries, not different
magnitudes.

ERA PIN. The v8 arms load only under the v8-era code (``b13b30b``); this file therefore runs
from an era-pinned worktree for ``--family v8`` and from the current tree for ``--family gen``.
The v8 era's rust bridge predates the seedless-seed fix (``bc00d4d``), so ``--impl node`` is
mandatory there — a seedless rust START replayed ONE dice stream at that commit.

Run (v8 family, from the era worktree):
  PYTHONPATH=/tmp/probeP_v8era/src nice -n 15 python <this> --family v8 --shard 0/2
Run (gen family, from the current tree):
  nice -n 15 python <this> --family gen --impl rust
  (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
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

from agents.gen3_data import moves as gen3_moves  # noqa: E402
from agents.gen3_data import type_chart as gen3_types  # noqa: E402
from agents.inference.player import RLPlayer  # noqa: E402
from agents.model.snapshot import current_model_version, load_foreign_opponent  # noqa: E402
from agents.observation.state_encoder import load_mappings  # noqa: E402
from utils.bridge.local_battle_runner import run_local_battles  # noqa: E402
from utils.team_loader import TeamLoader  # noqa: E402
from utils.teambuilder import Gen3Teambuilder  # noqa: E402

MAIN = "/home/goodlad/dev/gen3ai"
MD = f"{MAIN}/models"

# ---------------------------------------------------------------------------------------------
# The two arm FAMILIES. Both are (parent, fold, fixed reference opponent) triples where the
# reference is an ANCESTOR OF BOTH and equal to neither — using the parent as opponent would
# make the parent arm a self-mirror.
# ---------------------------------------------------------------------------------------------
FAMILIES = {
    # v8: probe P's exact arms. `final_model_interrupted.zip` is what those runs wrote.
    "v8": {
        "parent": (f"{MD}/ai_v8_04_distill_4teacher_0722/final_model_interrupted.zip",
                   f"{MD}/ai_v8_04_distill_4teacher_0722/model_config.json"),
        "fold": (f"{MD}/ai_v8_14_distill3_0725/final_model_interrupted.zip",
                 f"{MD}/ai_v8_14_distill3_0725/model_config.json"),
        "ref": (f"{MD}/ai_v8_03_zarch_control_0718/final_model_interrupted.zip",
                f"{MD}/ai_v8_03_zarch_control_0718/model_config.json"),
    },
    # gen: the rev-3 fold that did NOT gift (probe Q measured −0.75pp on untaught teams),
    # against its own parent, with rev-1 as the shared ancestor reference.
    "gen": {
        "parent": (f"{MD}/ai_v9_59_R2ACTION_0827/final_model.zip",
                   f"{MD}/ai_v9_59_R2ACTION_0827/model_config.json"),
        "fold": (f"{MD}/ai_v9_70_R3ACTION_0828/final_model.zip",
                 f"{MD}/ai_v9_70_R3ACTION_0828/model_config.json"),
        "ref": (f"{MD}/ai_v9_29_rev1_0823/final_model.zip",
                f"{MD}/ai_v9_29_rev1_0823/model_config.json"),
    },
}

# Probe P's pre-registered selection (16 untaught probe teams, 6 taught controls, 8 fixed
# opponent teams), copied verbatim from `/tmp/probeP/selection.json` so this file is
# self-contained and the cell identity is auditable without that scratch dir.
V8_SELECTION = {
    "probe_untaught": ["d0a4d2bcb8", "c90e782cad", "a6b630e6b4", "a577a735b7",
                       "9292a21833", "eaa88395e7", "7c2cb5cec1", "89fcef3b53",
                       "32f549483f", "593d7fb8a8", "7163ad9387", "dd460484fc",
                       "b26ed9c8e1", "c84f2b64a2", "f593373169", "048182d1e9"],
    "control_taught": ["564b9be3ae", "7594a34f82", "044da80d78",
                       "5c88ff9ca5", "4771662cf7", "45995e432f"],
    "opponents": ["c1fc379c85", "8a29df031e", "6c129cb50c", "511c359e2e",
                  "0af19b638a", "c0e60903e8", "2c5b1d6abb", "f7e46432b9"],
    "labels": {
        "d0a4d2bcb8": "balance", "c90e782cad": "balance", "a6b630e6b4": "balance",
        "a577a735b7": "balance", "9292a21833": "hyper_offense", "eaa88395e7": "hyper_offense",
        "7c2cb5cec1": "hyper_offense", "89fcef3b53": "offense", "32f549483f": "offense",
        "593d7fb8a8": "offense", "7163ad9387": "semi_stall", "dd460484fc": "semi_stall",
        "b26ed9c8e1": "semi_stall", "c84f2b64a2": "stall", "f593373169": "stall",
        "048182d1e9": "stall", "564b9be3ae": "semi_stall", "7594a34f82": "semi_stall",
        "044da80d78": "stall", "5c88ff9ca5": "stall", "4771662cf7": "stall",
        "45995e432f": "stall",
    },
}

# ---------------------------------------------------------------------------------------------
# Action classification. Sourced from the LegalActions snapshot + gen3_data move facts.
# ---------------------------------------------------------------------------------------------
CLASSES = ("SWITCH", "ATTACK", "SETUP", "RECOVER", "STATUS", "PROTECT", "PHAZE", "HAZARD",
           "OTHER_STATUS", "STRUGGLE")


def move_class(mid: str) -> str:
    md = gen3_moves.get(mid)
    if md is None:
        return "OTHER_STATUS"
    if md.base_power and md.base_power > 0:
        return "ATTACK"
    if md.is_boost:
        return "SETUP"
    if md.is_heal:
        return "RECOVER"
    if md.is_protect:
        return "PROTECT"
    if md.is_phaze:
        return "PHAZE"
    if md.is_hazard:
        return "HAZARD"
    if md.status_inflicted:
        return "STATUS"
    return "OTHER_STATUS"


def _tname(t) -> str:
    """poke-env PokemonType / str → the type_chart's enum-name key."""
    n = getattr(t, "name", None) or str(t)
    return n.upper()


def effectiveness(att_type, def_types) -> float:
    m = 1.0
    a = _tname(att_type)
    for d in def_types:
        try:
            m *= gen3_types.multiplier(_tname(d), a)
        except KeyError:
            pass
    return m


def best_effectiveness(mon_types, def_types) -> float:
    """Best STAB effectiveness a mon of ``mon_types`` has into ``def_types`` — the model-free
    matchup proxy. Uses the mons' TYPES, not their movesets, so it is symmetric and needs no
    knowledge of the opponent's four moves (which we usually do not have)."""
    if not mon_types or not def_types:
        return 1.0
    return max(effectiveness(t, def_types) for t in mon_types)


def classify_decision(legal, view) -> dict:
    """Everything about one decision that does NOT depend on which arm is choosing.

    Returns the per-action class vector, the board strata, and the derived per-action facts the
    conditional axes need (super-effectiveness, base power, switch-target resistance).
    """
    cls = [None] * 11
    bp = [0.0] * 11
    eff = [0.0] * 11
    sw_resist = [0.0] * 11
    ours = view.ours
    opp = view.opp
    oa = ours.active
    pa = opp.active
    opp_types = tuple(pa.types) if pa else ()
    our_types = tuple(oa.types) if oa else ()

    # switch targets live on OUR side; resolve their types for the resist axis
    by_species = {m.species: m for m in ours.mons}
    for sw in legal.switches:
        if not (0 <= sw.slot < 6):
            continue
        cls[sw.slot] = "SWITCH"
        tgt = by_species.get(sw.species)
        if tgt is not None and opp_types:
            # "does the incoming mon resist what the opponent is": best STAB effectiveness of
            # the OPPONENT'S types into the switch target. < 1.0 ⇒ a resisted pivot.
            sw_resist[sw.slot] = 1.0 if best_effectiveness(opp_types, tgt.types) < 1.0 else 0.0

    for i, m in enumerate(legal.move_slots[:4]):
        if m.disabled:
            continue
        a = 6 + i
        cls[a] = move_class(m.id)
        md = gen3_moves.get(m.id)
        if md is not None:
            bp[a] = float(md.base_power or 0)
            if md.base_power and opp_types:
                eff[a] = effectiveness(md.type, opp_types)
    if legal.struggle:
        cls[10] = "STRUGGLE"

    # ---- board strata (model-free) ----
    our_alive = sum(1 for m in ours.mons if not m.fainted)
    opp_alive = opp.team_size - sum(1 for m in opp.mons if m.fainted)
    # matchup sign: our best STAB into them, vs theirs into us
    us_into_them = best_effectiveness(our_types, opp_types) if our_types and opp_types else 1.0
    them_into_us = best_effectiveness(opp_types, our_types) if our_types and opp_types else 1.0
    if them_into_us > us_into_them:
        matchup = "losing"
    elif us_into_them > them_into_us:
        matchup = "winning"
    else:
        matchup = "even"

    return {
        "cls": cls, "bp": bp, "eff": eff, "sw_resist": sw_resist,
        "turn": int(view.turn),
        "force_switch": bool(legal.force_switch),
        "trapped": bool(legal.trapped),
        "n_sw": len(legal.switches),
        "n_mv": sum(1 for m in legal.move_slots[:4] if not m.disabled),
        "our_hp": float(oa.hp_fraction) if oa else 0.0,
        "opp_hp": float(pa.hp_fraction) if pa else 0.0,
        "our_alive": int(our_alive),
        "opp_alive": int(opp_alive),
        "matchup": matchup,
        "our_status": bool(oa.status) if oa else False,
        "our_boosted": bool(oa and any(v > 0 for v in oa.boosts.values())),
    }


# ---------------------------------------------------------------------------------------------
# The dual-scoring player
# ---------------------------------------------------------------------------------------------
class DualScoringPlayer(RLPlayer):
    """Acts with ``model``; scores ``other`` on the SAME (obs, mask) at every decision.

    The extra forward is the whole instrument: it is what makes the comparison an
    IDENTICAL-BOARD comparison rather than a comparison of two different games.
    """

    def __init__(self, *a, other=None, sink=None, meta=None, **kw):
        super().__init__(*a, **kw)
        self._other = other
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
        if self._sink is None:  # --outcomes-only: play the cells, record nothing per decision
            return RLPlayer.choose_move(self, battle)
        self._pending = None
        order = super().choose_move(battle)
        row, self._pending = self._pending, None
        # A DEFAULT order means the decision was NOT the model's (no legal action, or the
        # re-decide budget was exhausted); `DefaultBattleOrder.order` is the non-None string
        # `/choose default`, so the identity test has to be on the TYPE, not on truthiness.
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

        act_logits = th.as_tensor(snap["logits"])[None, :] + neg
        act_p = th.softmax(act_logits, dim=1)[0].numpy()
        with th.no_grad():
            od = self._other.policy.get_distribution(pin)
            oth_logits = od.distribution.logits + neg
            oth_p = th.softmax(oth_logits, dim=1)[0].numpy()
            oth_v = float(self._other.policy.predict_values(pin)[0].item())
        oth_idx = int(np.argmax(np.where(m > 0, oth_logits[0].numpy(), -1e30)))

        ctx = self._get_tracker(battle).last_ctx
        d = classify_decision(ctx.legal, battle.live_view())
        d.update(self._meta)
        d["tag"] = battle.strict_view().battle_tag
        d["act_idx"] = int(idx)
        d["oth_idx"] = oth_idx
        d["act_v"] = float(snap["value"])
        d["oth_v"] = oth_v
        d["act_p"] = [round(float(x), 5) for x in act_p]
        d["oth_p"] = [round(float(x), 5) for x in oth_p]
        d["our_hp"] = round(d["our_hp"], 4)
        d["opp_hp"] = round(d["opp_hp"], 4)
        d["act_v"] = round(d["act_v"], 5)
        d["oth_v"] = round(d["oth_v"], 5)
        return d


_ACCT = itertools.count(1)


def _acct(tag: str) -> AccountConfiguration:
    return AccountConfiguration(f"M4{tag[:2]}{next(_ACCT):05d}", "pw")


def load(zip_path: str, cfg: str, cv):
    m, _ = load_foreign_opponent(zip_path, current_version=cv, device="cpu", config_path=cfg)
    fe = m.policy.features_extractor
    if hasattr(fe, "_debugger"):
        fe._debugger = None
    m.policy.set_training_mode(False)
    return m


def acid_test(models: dict) -> dict:
    """Three facts that would invalidate everything downstream if false: the arms LOAD, they are
    DISTINCT networks in parameter space, and the fold sits nearer its parent than the reference
    (the lineage order). A mis-resolved path that loads one zip twice reads as a perfect null."""
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
    ok = all(v > 1e-3 for v in pmat.values())
    d_fp = pmat.get("fold|parent", pmat.get("parent|fold", 0.0))
    d_fr = pmat.get("fold|ref", pmat.get("ref|fold", 0.0))
    return {"pairwise_param_l2": pmat, "all_distinct": bool(ok),
            "fold_nearer_parent_than_ref": bool(d_fp < d_fr),
            "n_params": {n: int(sum(p.numel() for p in m.policy.parameters()))
                         for n, m in models.items()}}


def sha10(s: str) -> str:
    return hashlib.sha1(s.strip().encode()).hexdigest()[:10]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="v8", choices=sorted(FAMILIES))
    ap.add_argument("--games", type=int, default=8, help="battles per (team, opp, arm) cell")
    ap.add_argument("--opps", type=int, default=4, help="how many of the 8 fixed opponents")
    ap.add_argument("--untaught", type=int, default=16)
    ap.add_argument("--taught", type=int, default=6)
    ap.add_argument("--impl", default="node", choices=("node", "rust"))
    ap.add_argument("--shard", default="0/1", help="i/k over the probe-team list")
    ap.add_argument("--out", default="/tmp/m4/rows")
    ap.add_argument("--teams-json", default="", help="explicit selection json (gen family)")
    ap.add_argument("--outcomes-only", action="store_true",
                    help="replay the SAME cells recording only per-game outcomes. Greedy play on "
                         "a fixed seed is deterministic, so the outcomes join to an earlier "
                         "dual-scored pass battle-for-battle — this is how a run made before the "
                         "per-game vectors existed gets its battle-level attribution without "
                         "re-recording 40k decision rows.")
    a = ap.parse_args(argv)

    fam = FAMILIES[a.family]
    if a.family == "v8" and a.impl != "node":
        raise SystemExit("[m4] the v8 era predates the rust seedless-seed fix — node is mandatory")

    t0 = time.time()
    mappings = load_mappings()
    cv = current_model_version(mappings)
    models = {k: load(*fam[k], cv) for k in ("parent", "fold", "ref")}
    acid = acid_test(models)
    print(f"[m4] ACID {json.dumps(acid)}", flush=True)
    if not acid["all_distinct"]:
        raise SystemExit("[m4] ACID FAILED — arms are not distinct networks")

    if a.teams_json:
        sel = json.load(open(a.teams_json))
    else:
        sel = V8_SELECTION
    pool = {sha10(t): t for t in TeamLoader().get_all_teams()}
    missing = [s for s in sel["probe_untaught"] + sel["control_taught"] + sel["opponents"]
               if s not in pool]
    if missing:
        raise SystemExit(f"[m4] selection GIGO: {missing} not in the 719-team pool")

    probes = ([(s, "untaught") for s in sel["probe_untaught"][:a.untaught]]
              + [(s, "taught") for s in sel["control_taught"][:a.taught]])
    si, sk = (int(x) for x in a.shard.split("/"))
    probes = [p for i, p in enumerate(probes) if i % sk == si]
    opps = sel["opponents"][:a.opps]
    print(f"[m4] {len(probes)} probe teams x {len(opps)} opps x {a.games}g x 2 arms = "
          f"{len(probes) * len(opps) * a.games * 2} battles", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tag = f"{a.family}_s{si}"
    sink = None if a.outcomes_only else gzip.open(f"{a.out}_{tag}.jsonl.gz", "ab")
    cellsink = open(f"{a.out}_{tag}_cells.jsonl", "a")

    n_rows = 0
    for team_sha, kind in probes:
        team_str = pool[team_sha]
        for opp_sha in opps:
            # CRN: probe P's own per-cell seed construction, so this probe's battles ARE
            # (a subset of) probe P's battles rather than a fresh draw.
            rng = random.Random(f"{team_sha}:{opp_sha}")
            seeds = [[rng.randrange(0, 65536) for _ in range(4)] for _ in range(a.games)]
            for arm in ("parent", "fold"):
                other = "fold" if arm == "parent" else "parent"
                ts = time.time()
                meta = {"team": team_sha, "kind": kind, "opp": opp_sha, "arm": arm,
                        "arch": sel["labels"].get(team_sha)}
                p = DualScoringPlayer(
                    model=models[arm], team=Gen3Teambuilder([team_str]),
                    battle_format="gen3ou", server_configuration=LocalhostServerConfiguration,
                    mappings=mappings, account_configuration=_acct(arm),
                    stochastic=False, start_listening=False,
                    other=None if a.outcomes_only else models[other], sink=sink, meta=meta)
                o = RLPlayer(model=models["ref"], team=Gen3Teambuilder([pool[opp_sha]]),
                             battle_format="gen3ou",
                             server_configuration=LocalhostServerConfiguration,
                             mappings=mappings, account_configuration=_acct("rf"),
                             stochastic=False, start_listening=False)
                # PER-GAME outcomes + the battle tag they belong to. Both arms play the same
                # seed list, so game index i is the SAME battle for both — which is what makes a
                # battle-level `fold won where the parent lost` attribution possible at all.
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
                rec = {"team": team_sha, "kind": kind, "opp": opp_sha, "arm": arm,
                       "arch": sel["labels"].get(team_sha),
                       "wins": p.n_won_battles, "finished": p.n_finished_battles,
                       "requested": a.games, "rows": p.n_rows,
                       # The two deferral counters. `n_redecides > 0` is the only path on which
                       # the recorder could see a SUPERSEDED decision, so recording them is what
                       # lets the write-guard's correctness be a MEASUREMENT rather than a claim.
                       "n_defaults": p._n_defaults, "n_redecides": p._n_redecides,
                       "n_decisions": p._n_decisions,
                       "per_game": per_game, "tags": tags,
                       "secs": round(time.time() - ts, 1)}
                cellsink.write(json.dumps(rec) + "\n")
                cellsink.flush()
                if sink is not None:
                    sink.flush()
                n_rows += p.n_rows
                print(f"  {kind:8s} {team_sha} vs {opp_sha} [{arm:6s}] "
                      f"{p.n_won_battles}/{p.n_finished_battles} rows={p.n_rows} "
                      f"{rec['secs']:.0f}s (elapsed {time.time() - t0:.0f}s)", flush=True)
    if sink is not None:
        sink.close()
    cellsink.close()
    print(f"[m4] done: {n_rows} decision rows in {time.time() - t0:.0f}s", flush=True)
    with open(f"{a.out}_{tag}_meta.json", "w") as f:
        json.dump({"acid": acid, "family": a.family, "games": a.games, "opps": opps,
                   "shard": a.shard, "impl": a.impl, "n_rows": n_rows,
                   "wall_s": round(time.time() - t0, 1)}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
