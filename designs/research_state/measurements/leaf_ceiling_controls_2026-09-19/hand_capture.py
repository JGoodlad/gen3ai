"""CONTROL 1 — capture the OMNISCIENT board at each banked fork's successor, so a HAND EVALUATOR
can be scored on the EXACT states arm W's head was scored on.

    python3 hand_capture.py --forks <banked fork dir> --snapshot <zip> --out <dir> --shard i/W

WHY A RE-RUN AT ALL. The banked fork sets store the successor OBSERVATION (a 2501-dim one-sided
vector) and nothing else. A hand evaluator wants the BOARD — both sides' HP, alive counts, status,
boosts, hazards — and the omniscient variant wants the opponent's TRUE bench, which by construction
is not in our obs. So each banked branch is replayed under the SAME common random numbers (the
record's own resolved start seed, both sides greedy, the substitute action at the same turn), and at
the first LIVE decision the board is read off BOTH players' battle objects at once:

  * OUR side, from the trainee's own ``LiveView``            → the ONE-SIDED readout
  * the opponent's side, from the OPPONENT player's own      → the OMNISCIENT readout
    ``LiveView`` (in ITS view, ``live.ours`` is its full team)

Both players are in-process (``run_local_battles``), so this is one instant of one battle, not two
runs stitched together. ``opp_turn`` is recorded so a reader can see the two views were at the same
turn (they are asserted equal and counted when not).

THE ALIGNMENT IS PROVEN, NOT ASSUMED. The re-run recomputes the successor obs with the banked
builder's own capture discipline and compares it to ``succ_<tag>.npy[local]`` element-wise. A
mismatch means either the CRN did not reproduce or the row→successor index is wrong, and both are
fatal to this read, so the rate is reported per shard and every row carries its own ``obs_match``.
That is a strictly stronger check than re-running the OUTCOME: the outcome is one bit.

WHY IT FORFEITS. Nothing here needs the branch's outcome — the banked rows already carry it, under
the identical dice. So the trainee returns a ``ForfeitBattleOrder`` on the decision AFTER the
capture, which ends the line at ~turn T+1 instead of ~turn 55. ``--to-terminal N`` disables that for
the first N forks of the shard and compares the realized outcome against the banked one, so the
"the re-run is the same line" claim is checked rather than argued.

Output, per shard: ``hand_<tag>.jsonl`` — one line per captured successor, carrying the LOCAL
successor index (so it joins to ``succ_<tag>.npy`` row-for-row) and the raw board quantities. The
potentials themselves are computed by the PRODUCTION code (``agents.training.reward_potentials``),
not re-implemented here.

CPU only. Nothing is written under ``models/``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import traceback

import numpy as np

BRANCHES = ("top1", "top2", "rand")
_MEAS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_MEAS, "paired_refit_discrimination_2026-09-14"))


def _phi_shim():
    """A minimal object carrying the mixin's contract, so the PRODUCTION potentials run unmodified.

    ``RewardPotentials`` is a mixin over ``Gen3RewardManager``; the six board potentials read only
    ``self.config`` (``mat_alive_weight``), ``self.progress_clock`` and ``self._belief_memo``, and
    write ``self._last_material_margin``. Building the real manager would drag in an env, a config
    parse and an episode lifecycle none of which this read has or wants."""
    from agents.observation.incoming_damage_encoder import IncomingBeliefMemo
    from agents.training.reward_bias_terms import RewardBiasTerms
    from agents.training.reward_config import RewardConfig
    from agents.training.reward_potentials import RewardPotentials

    class _Shim(RewardPotentials):
        # `_compute_phi_hazard` reaches `self._opp_spikes`, which `Gen3RewardManager` resolves to
        # RewardBiasTerms'. BORROWED, never re-implemented — a second copy of a one-line accessor
        # is exactly how a control drifts from the thing it is controlling.
        _opp_spikes = RewardBiasTerms._opp_spikes
        def __init__(self):
            self.config = RewardConfig()
            self.progress_clock = None
            self._belief_memo = IncomingBeliefMemo()
            self._last_material_margin = 0.0

    return _Shim()


_TEMPO = frozenset({"par", "slp", "frz"})


def _side_raw(side, team_size_default=6):
    """The raw board quantities for ONE LiveSide, from that side's OWNER's point of view (so an
    own-side read is complete and an opponent-side read is what we have SEEN)."""
    mons = list(side.mons[:6])
    hp = sum(float(m.hp_fraction) for m in mons if not m.fainted)
    alive = sum(1 for m in mons if not m.fainted)
    tempo = sum(1 for m in mons if not m.fainted and m.status in _TEMPO)
    statused = sum(1 for m in mons if not m.fainted and m.status is not None)
    declared = int(getattr(side, "team_size", None) or team_size_default)
    active = side.active
    return {
        "n_known": len(mons),
        "declared": min(declared, 6),
        "hp_known": hp,
        "alive_known": alive,
        "tempo": tempo,
        "statused": statused,
        "spikes": int((side.side_conditions or {}).get("spikes", 0)),
        "active_species": None if active is None else active.species,
        "active_hp": 0.0 if active is None else float(active.hp_fraction),
        "active_pos_boosts": 0 if active is None
        else sum(max(0, int(v)) for v in (active.boosts or {}).values()),
        "active_neg_boosts": 0 if active is None
        else sum(min(0, int(v)) for v in (active.boosts or {}).values()),
        "active_types": [] if active is None else [t for t in (active.types or ()) if t],
        "alive_types": [[t for t in (m.types or ()) if t] for m in mons if not m.fainted],
    }


class _Capture:
    """Wrap the trainee's ``choose_move``: record the FIRST LIVE decision's obs + both boards, then
    (optionally) forfeit on the decision after it."""

    def __init__(self, player, opponent, *, forfeit: bool):
        self.player = player
        self.opponent = opponent
        self.forfeit = forfeit
        self.obs = None
        self.mask = None
        self.board = None
        self.err = None
        self._orig = player.choose_move

        def wrapped(battle):
            if self.obs is None:
                try:
                    tracker = player._get_tracker(battle)
                    snap = tracker.snapshot()
                    d = player.embed_battle(battle)
                    tracker.restore(snap)
                    m = np.asarray(d["action_mask"]).reshape(-1)
                    if int(m.sum()) > 0:
                        self.obs = np.asarray(d["observation"], dtype=np.float32)
                        self.mask = m.astype(np.int8)
                        self.board = self._read_boards(battle)
                except Exception as e:                       # noqa: BLE001
                    self.err = f"{type(e).__name__}: {e}"
            elif self.forfeit:
                from poke_env.player.battle_order import ForfeitBattleOrder
                return ForfeitBattleOrder()
            return self._orig(battle)

        player.choose_move = wrapped

    def _read_boards(self, battle):
        ours = battle.strict_view().live
        opp_battle = None
        for b in list(getattr(self.opponent, "_battles", {}).values()):
            if not getattr(b, "finished", False):
                opp_battle = b
        theirs = opp_battle.strict_view().live if opp_battle is not None else None
        out = {
            "turn": int(getattr(battle, "turn", 0) or 0),
            "opp_turn": None if opp_battle is None else int(getattr(opp_battle, "turn", 0) or 0),
            "our": _side_raw(ours.ours),
            "opp_seen": _side_raw(ours.opp),
            # the opponent's OWN view of its OWN side — the omniscient half
            "opp_true": None if theirs is None else _side_raw(theirs.ours),
            "our_by_opp": None if theirs is None else _side_raw(theirs.opp),
        }
        shim = _phi_shim()
        out["phi"] = {
            "mat": float(shim._compute_phi_mat(ours)),
            "status": float(shim._compute_phi_status(ours)),
            "hazard": float(shim._compute_phi_hazard(ours)),
            "boost": float(shim._compute_phi_boost(ours)),
            "opp_boosts": float(shim._compute_phi_opp_boosts(ours)),
            "roar": float(shim._compute_phi_roar(ours)),
        }
        try:
            phi_b, risk, minb = shim._belief_potential_and_risk(ours)
            out["phi"]["belief"] = float(phi_b)
            out["phi"]["active_risk"] = float(risk)
            out["phi"]["min_bench_pko"] = float(minb)
        except Exception as e:                               # noqa: BLE001
            out["phi"]["belief_error"] = f"{type(e).__name__}: {e}"
        return out


def run_branch(record, trace, turn, action, *, play_model, opp_model, opp_ckpt, mappings, impl, tag,
               forfeit):
    """One branch, replayed under the record's own dice, captured at the first live decision."""
    from poke_env.ps_client import LocalhostServerConfiguration
    from main.prober.replay import build_opponent, build_trainee
    from utils.bridge.counterfactual import replay_counterfactual as _run_one

    choice_map = trace.action_choices or {}
    if int(action) not in choice_map:
        return None, None
    trainee = build_trainee(play_model, record, mappings, LocalhostServerConfiguration, tag=tag)
    opponent, _src = build_opponent(
        "", record, play_model, mappings, LocalhostServerConfiguration,
        opponent_ckpt=opp_ckpt, opp_model=opp_model, opponent_source="ckpt",
        opponent_stochastic=False, tag=tag)
    cap = _Capture(trainee, opponent, forfeit=forfeit)
    res = _run_one(record, trainee=trainee, opponent=opponent, divergence_turn=int(turn),
                   substitute_choice=choice_map[int(action)],
                   seed=record.start_options().get("seed"), post_t_seed=None, impl=impl)
    return cap, res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forks", required=True, help="the BANKED fork directory")
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1", help="i/W — which forks_s*.jsonl slice")
    ap.add_argument("--max-forks", type=int, default=0)
    ap.add_argument("--to-terminal", type=int, default=6,
                    help="run the first N forks of the shard to a TERMINAL (no forfeit) and check "
                         "the realized outcome against the banked one")
    ap.add_argument("--impl", default="rust", choices=["node", "rust"])
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args(argv)

    import torch
    torch.set_num_threads(args.threads)
    from agents.observation.state_encoder import load_mappings
    from agents.training.obs_materializer import materialize_from_record
    from utils.bridge.reconstruction import ReconstructionRecord
    from forks import load_model                      # by path, unmodified

    os.makedirs(args.out, exist_ok=True)
    i, w = (int(x) for x in args.shard.split("/"))
    t0 = time.time()

    # -- the banked shard, read with load_forks' OWN drop rule (succ past the last npy flush) ----
    paths = sorted(glob.glob(os.path.join(args.forks, "forks_s*.jsonl")))
    if not (0 <= i < len(paths)) or w != len(paths):
        raise SystemExit(f"REFUSED: --shard {args.shard} does not name one of {len(paths)} banked "
                         f"shard files; this control is aligned to the BANKED sharding")
    rp = paths[i]
    tag = os.path.basename(rp)[len("forks_"):-len(".jsonl")]
    S = np.load(os.path.join(args.forks, f"succ_{tag}.npy"))
    rows = []
    for line in open(rp):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            break
        if any((r["branches"][b].get("succ") is not None
                and r["branches"][b]["succ"] >= len(S)) for b in BRANCHES):
            continue
        rows.append(r)
    if args.max_forks:
        rows = rows[: args.max_forks]
    print(f"[hand {tag}] {len(rows)} banked forks, {len(S)} banked successors", flush=True)

    meta_path = os.path.join(args.forks, f"meta_{tag}.json")
    bmeta = json.load(open(meta_path))
    tree = bmeta["tree"]
    plan = json.load(open(os.path.join(tree, "plan.json")))
    sent = {it["key"]: it.get("path") for it in plan["items"] if it.get("kind") == "sentinel"}

    play_model = load_model(args.snapshot)
    opp_models = {}
    mappings = load_mappings()

    out_path = os.path.join(args.out, f"hand_{tag}.jsonl")
    fout = open(out_path, "a")
    stats = {"forks": 0, "branches": 0, "captured": 0, "obs_match": 0, "obs_mismatch": 0,
             "no_capture": 0, "turn_mismatch": 0, "no_opp_view": 0, "errors": 0,
             "terminal_checks": 0, "terminal_outcome_agree": 0}
    for n, r in enumerate(rows):
        try:
            base = os.path.join(tree, r["opp"], r["base"])
            rec = ReconstructionRecord.load(base + "_reconstruction.json")
            npz = np.load(base + "_states.npz")
            actions = np.asarray(npz["actions"], dtype=int)
            trace = materialize_from_record(rec, actions=actions, mappings=mappings,
                                            map_actions_at=r["inv"],
                                            stop_after_decision=r["inv"], impl=args.impl)
            if r["opp"] not in opp_models:
                opp_models[r["opp"]] = load_model(sent[r["opp"]])
            to_terminal = n < args.to_terminal
            for name in BRANCHES:
                br = r["branches"][name]
                if br.get("succ") is None:
                    continue
                stats["branches"] += 1
                cap, res = run_branch(rec, trace, r["turn"], br["action"],
                                      play_model=play_model, opp_model=opp_models[r["opp"]],
                                      opp_ckpt=sent[r["opp"]], mappings=mappings, impl=args.impl,
                                      tag=f"h{i}_{n}_{name}", forfeit=not to_terminal)
                if cap is None or cap.obs is None or cap.board is None:
                    stats["no_capture"] += 1
                    if stats["no_capture"] <= 5:
                        print(f"  no-capture {name}: cap={cap is not None} "
                              f"obs={None if cap is None else (cap.obs is not None)} "
                              f"board={None if cap is None else (cap.board is not None)} "
                              f"err={None if cap is None else cap.err}", flush=True)
                    continue
                stats["captured"] += 1
                match = bool(np.array_equal(cap.obs, S[br["succ"]]))
                stats["obs_match" if match else "obs_mismatch"] += 1
                bd = cap.board
                if bd.get("opp_true") is None:
                    stats["no_opp_view"] += 1
                elif bd.get("opp_turn") != bd.get("turn"):
                    stats["turn_mismatch"] += 1
                if to_terminal:
                    stats["terminal_checks"] += 1
                    if (res or {}).get("outcome") == br["outcome"]:
                        stats["terminal_outcome_agree"] += 1
                fout.write(json.dumps({
                    "succ": int(br["succ"]), "shard": tag, "fork_line": n, "branch": name,
                    "battle": r["base"], "inv": r["inv"], "fork_turn": r["turn"],
                    "obs_match": match, "board": bd,
                    "rerun_outcome": None if not to_terminal else (res or {}).get("outcome"),
                    "banked_outcome": br["outcome"],
                }) + "\n")
            stats["forks"] += 1
            if stats["forks"] % 20 == 0:
                fout.flush()
                el = time.time() - t0
                print(f"  [{tag}] {stats['forks']}/{len(rows)} forks ({el:.0f}s, "
                      f"{el/max(1,stats['forks']):.2f}s/fork) {stats}", flush=True)
        except Exception as e:                               # noqa: BLE001
            stats["errors"] += 1
            if stats["errors"] <= 5:
                print(f"  ERR {r['base']}#{r['inv']}: {type(e).__name__} {e}", flush=True)
                traceback.print_exc()
    fout.close()
    meta = {"shard": args.shard, "tag": tag, "forks_dir": args.forks, "tree": tree,
            "snapshot": args.snapshot, "stats": stats, "wall_s": time.time() - t0,
            "impl": args.impl, "to_terminal": args.to_terminal}
    json.dump(meta, open(os.path.join(args.out, f"handmeta_{tag}.json"), "w"), indent=1)
    print(f"[hand {tag}] DONE {json.dumps(stats)} in {meta['wall_s']:.0f}s", flush=True)
    if stats["obs_mismatch"]:
        print(f"🚨 [hand {tag}] {stats['obs_mismatch']} OBS MISMATCHES — the re-run did not "
              f"reproduce the banked successor; the join is NOT proven for those rows", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
