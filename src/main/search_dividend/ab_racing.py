"""The offline A/B: does ADAPTIVE allocation beat the fixed grid at matched compute?

Two phases, and the split is the whole method.

**Phase 1 — BANK.** For each of N recorded decisions, draw ``R`` independent CRN-paired rounds (one
determinized world, one freshly minted dice seed, the α-weighted opponent marginalization inside)
and record the per-action value vector each round produced. The result is an ``R x A`` matrix per
decision, written to a JSON file. This is the only phase that touches the sim, the model or the
clock, and it is deliberately allocator-BLIND: it scores every action on every round, so neither
arm can have been favoured by how the samples were drawn.

**Phase 2 — REPLAY.** Both allocators are then run over that ONE bank. The grid consumes rounds
``0..n-1`` scoring everything; the racer consumes the same rounds in the same order but only its
live entries. At every budget point the two arms see IDENTICAL samples, so the comparison is
PAIRED at the sample level and not merely matched in expectation — the difference between them is
allocation and nothing else. It is also free: a whole budget sweep costs no sim time, so the curves
can be re-cut without re-measuring.

**The GOLD reference** is the grid's argmax over the full bank, with the doubling check the
registration asked for reported beside it (how many argmaxes move between ``R/2`` and ``R`` rounds).
A gold that is itself unstable is not a reference, and the honest thing is to publish the number
rather than assume it.

**Compute is priced in TWO units and they answer different questions.** *Arm evaluations* (one sim
branch + one obs materialization + its share of the batched critic forward) are what elimination
saves. *Model seconds* add the per-round ``open_root``, which racing pays MORE of — it runs more,
cheaper rounds — so the arm-evaluation ratio flatters it and the seconds ratio is the honest
headline. Both are reported; the cost constants come from the banking run's own measurement.

    # phase 1 (slow, needs the sim + a current-architecture checkpoint)
    python -m main.search_dividend.ab_racing <ckpt_or_run> --traces <eval_traces_dir> \
        --decisions 120 --rounds 32 --bank tmp/racing_bank.json

    # phase 2 (pure, seconds)
    python -m main.search_dividend.ab_racing --replay tmp/racing_bank.json --out report.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from main.search_dividend.alpha import build_candidates, legal_choices_from_request
from main.search_dividend.budget import Deadline, RealizedWidths, WidthCaps
from main.search_dividend.racing import RULES, Racer, RacingConfig, grid_over_bank
from main.search_dividend.search import SearchConfig, SearchEngine, _PlyContext

#: Agreement levels the report solves the budget-to-reach question at.
AGREEMENT_TARGETS = (0.80, 0.90, 0.95)

#: The two compute axes the report solves the budget-to-reach question ON, and they are NOT
#: interchangeable. ``budget_s`` is the DEADLINE a decision would have to be granted; ``spend_s`` is
#: what it actually consumed. They coincide for the grid, which always spends its whole budget, and
#: diverge for racing, which stops the moment the field collapses and hands the rest of the clock
#: back. Reporting only the first understates racing by exactly the unspent remainder; reporting
#: only the second flatters it, because a deadline still has to be reserved.
COST_AXES = ("budget_s", "spend_s")


# ---------------------------------------------------------------------------
# phase 1 — banking
# ---------------------------------------------------------------------------


@dataclass
class BankedDecision:
    """One decision's ``R x A`` paired sample matrix, plus what it takes to read it."""

    battle: str
    inv: int
    turn: int
    actions: List[int]
    policy_action: int
    #: ``rounds[i][j]`` is round *i*'s value for ``actions[j]``. A round is kept only when EVERY
    #: action produced a value — a partial round would make the two allocators see different
    #: samples, which is the one thing this design exists to prevent.
    rounds: List[List[float]] = field(default_factory=list)
    score_mode: str = ""
    rounds_dropped: int = 0

    def as_dict(self) -> dict:
        return asdict(self)

    def bank(self) -> List[Dict[int, float]]:
        return [{a: row[j] for j, a in enumerate(self.actions)} for row in self.rounds]


def bank_decision(engine: SearchEngine, *, record, side: str, turn: int,
                  our_history: Sequence[int], our_tokens: Dict[int, str],
                  observed_our_lines: Sequence[str], pub, rounds: int, m_opp: int,
                  opp_true_packed: Optional[str] = None
                  ) -> Tuple[List[Dict[int, float]], dict]:
    """Draw ``rounds`` CRN-paired samples, scoring EVERY action on each.

    This is :meth:`SearchEngine._run_racing`'s round loop with the elimination removed and nothing
    else changed — same world supply, same per-round fresh CRN seed, same depth-1 scoring, same
    prefix gate, same incomplete-round discard. ``racing_bank_matches_the_engines_round_loop`` in
    the test module pins that equivalence rather than leaving it to this docstring.
    """
    other = "p2" if side == "p1" else "p1"
    ss = engine.session()
    actions = sorted(our_tokens)
    worlds = engine._worlds(record, other, observed_our_lines, max(1, rounds), opp_true_packed)
    out: List[Dict[int, float]] = []
    widths = RealizedWidths(planned={}, n_our_actions=len(actions))
    deadline = Deadline(1e9)
    meta = {"dropped": 0, "gate_failed": 0, "open_failed": 0, "opens": 0, "score_mode": ""}
    for j in range(rounds):
        wrec, wmeta = worlds[j % len(worlds)]
        meta["opens"] += 1
        t_open = time.monotonic()
        try:
            root = ss.open_root(turn, record=wrec)
        except Exception as e:                       # noqa: BLE001
            widths.open_s += time.monotonic() - t_open
            meta["open_failed"] += 1
            meta["open_error"] = f"{type(e).__name__}: {e}"
            continue
        widths.open_s += time.monotonic() - t_open
        prefix = root.prefix_p1_chunks if side == "p1" else root.prefix_p2_chunks
        if not _prefix_ok(observed_our_lines, prefix, turn):
            meta["gate_failed"] += 1
            continue
        cands, _diag = build_candidates(
            legal_choices_from_request((root.requests or {}).get(other)), pub, m_opp=m_opp)
        if not cands:
            meta["dropped"] += 1
            continue
        ctx = _PlyContext(side=side, other=other, record=record, prefix=prefix,
                          our_history=list(our_history), pub=pub,
                          seeds=[engine._crn_seed(turn, j)], m_opp=m_opp)
        got = engine._score_world({"root": root, "meta": dict(wmeta), "cands": cands},
                                  ctx, dict(our_tokens), actions, widths, deadline, max_depth=1)
        if got is None:
            meta["dropped"] += 1
            continue
        per_action, _beam, adiag = got
        if not set(adiag.get("valued") or ()) >= set(actions):
            meta["dropped"] += 1
            continue
        meta["score_mode"] = adiag.get("score_mode") or meta["score_mode"]
        out.append({a: float(per_action[a]) for a in actions})
    meta["open_s"] = round(widths.open_s, 3)
    meta["arms"] = widths.arms_expanded
    # 🚨 The cost model's unit is an ACTION SLOT (one of our actions, on one round), not a sim
    # ARM (an action x an opponent candidate). `Price.seconds` multiplies by the racer's
    # `arms_spent`, which counts live ACTIONS per round — so pricing per sim arm would understate
    # a round by exactly `m_opp` and make every budget point 4x too generous. The α-marginalization
    # width is a CONSTANT multiplier both allocators pay identically, so it belongs inside the
    # constant rather than in the count.
    meta["slots"] = widths.arms_expanded // max(1, m_opp)
    return out, meta


def _prefix_ok(observed, prefix, turn) -> bool:
    from main.search_dividend import determinize as dz
    return dz.prefix_matches(observed, prefix, turn=turn)


# ---------------------------------------------------------------------------
# phase 2 — replay
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Price:
    """What a round and an arm cost, in seconds. Measured during banking, not assumed."""

    open_s: float = 0.06
    arm_s: float = 0.01

    def seconds(self, rounds: int, arms: int) -> float:
        return rounds * self.open_s + arms * self.arm_s


def _grid_under_seconds(bank, actions, budget_s: float, price: Price, prefer: int):
    """The grid at a wall-clock budget: as many FULL rounds as fit."""
    per_round = price.seconds(1, len(actions))
    n = min(len(bank), int(budget_s // per_round) if per_round > 0 else len(bank))
    if n <= 0:
        return None, 0, 0
    out = grid_over_bank(bank, actions, max_rounds=n, prefer=prefer)
    return out.action, out.rounds, out.arms_spent


def _race_under_seconds(bank, actions, budget_s: float, price: Price, cfg: RacingConfig,
                        prefer: int):
    """The racer at the same wall-clock budget: rounds while the clock affords the LIVE set."""
    r = Racer(actions, cfg)
    spent = 0.0
    for row in bank:
        if r.resolved():
            break
        cost = price.seconds(1, len(r.live))
        if spent + cost > budget_s:
            break
        spent += cost
        r.observe({a: row[a] for a in r.live})
    if r.rounds <= 0:
        return None, 0, 0
    return r.leader(prefer), r.rounds, r.arms_spent


def replay(banked: Sequence[BankedDecision], *, price: Price, cfg: RacingConfig,
           budgets_s: Sequence[float]) -> dict:
    """Both allocators over one bank, at every budget point. Pure and fast."""
    gold: Dict[int, int] = {}
    gold_means: Dict[int, Dict[int, float]] = {}
    half_moved = 0
    usable: List[BankedDecision] = []
    for i, d in enumerate(banked):
        if len(d.rounds) < 4 or len(d.actions) < 3:
            continue
        b = d.bank()
        full = grid_over_bank(b, d.actions, prefer=d.policy_action)
        half = grid_over_bank(b, d.actions, max_rounds=len(b) // 2, prefer=d.policy_action)
        gold[i] = full.action
        gold_means[i] = full.means
        half_moved += int(half.action != full.action)
        usable.append(d)
    idx = [i for i in gold]
    n = len(idx)

    curves = {"grid": [], "racing": []}
    for bs in budgets_s:
        for arm in ("grid", "racing"):
            agree = 0
            scored = 0
            regret: List[float] = []
            rounds_tot = arms_tot = 0
            for i in idx:
                d = banked[i]
                b = d.bank()
                if arm == "grid":
                    act, rnds, arms = _grid_under_seconds(b, d.actions, bs, price,
                                                          d.policy_action)
                else:
                    act, rnds, arms = _race_under_seconds(b, d.actions, bs, price, cfg,
                                                          d.policy_action)
                if act is None:
                    continue
                scored += 1
                rounds_tot += rnds
                arms_tot += arms
                agree += int(act == gold[i])
                m = gold_means[i]
                regret.append(float(m[gold[i]] - m[act]))
            curves[arm].append({
                "budget_s": round(bs, 4),
                "n": scored,
                "agreement": round(agree / scored, 4) if scored else None,
                "mean_regret": round(float(np.mean(regret)), 6) if regret else None,
                # The COST of the disagreements, not their count. Two arms can disagree with gold
                # equally often and be worth very different amounts: an allocator that errs only on
                # near-ties is a different object from one that errs on the decisive ones, and an
                # agreement rate alone cannot tell them apart.
                "mean_regret_on_disagree": (round(float(np.mean([r for r in regret if r > 0])), 6)
                                            if any(r > 0 for r in regret) else 0.0),
                "n_disagree": sum(1 for r in regret if r > 0),
                "mean_rounds": round(rounds_tot / scored, 2) if scored else None,
                "mean_arms": round(arms_tot / scored, 2) if scored else None,
                "spend_s": (round(price.seconds(rounds_tot, arms_tot) / scored, 4)
                            if scored else None),
            })

    return {
        "n_decisions": n,
        "n_banked": len(banked),
        "gold": {"rounds_mean": round(float(np.mean([len(d.rounds) for d in usable])), 2) if usable else 0,
                 "doubling_check_moved": half_moved,
                 "doubling_check_frac": round(half_moved / n, 4) if n else None},
        "price": asdict(price),
        "racing_cfg": cfg.as_dict(),
        "curves": curves,
        "budget_ratio": _budget_ratios(curves, "budget_s"),
        "spend_ratio": _budget_ratios(curves, "spend_s"),
        "separation": separation_profile(usable, cfg),
    }


def _budget_ratios(curves: dict, axis: str = "budget_s") -> dict:
    """Cost-to-reach-X%-agreement for each arm on ``axis``, and the ratio — the headline.

    Linear interpolation between the two bracketing budget points, because the swept grid is
    coarse and reporting the nearest swept point would quantize the answer to the sweep.
    ``None`` where an arm never reaches the target inside the swept range, which is a finding and
    not a gap: an arm that cannot get there has no budget to report.
    """
    def budget_for(rows, target):
        prev = None
        for row in rows:
            a = row["agreement"]
            if a is None or row.get(axis) is None:
                continue
            if a >= target:
                if prev is None or prev[1] >= target:
                    return row[axis]
                b0, a0 = prev
                f = (target - a0) / (a - a0) if a != a0 else 0.0
                return round(b0 + f * (row[axis] - b0), 4)
            prev = (row[axis], a)
        return None

    out = {}
    for t in AGREEMENT_TARGETS:
        g = budget_for(curves["grid"], t)
        r = budget_for(curves["racing"], t)
        out[f"{int(t*100)}%"] = {
            "grid_s": g, "racing_s": r,
            "reduction_x": round(g / r, 3) if (g and r) else None}
    return out


def separation_profile(banked: Sequence[BankedDecision], cfg: RacingConfig) -> dict:
    """WHERE the wins come from: how many rounds each decision needs to collapse to one action.

    The ``never`` mass is the regime in which racing saves nothing at all and a time manager should
    simply cap the decision — reporting it is the point of this function, because a mean
    samples-to-separation over the decisions that DID separate would hide it completely.
    """
    hist: Dict[str, int] = {}
    never = 0
    at: List[int] = []
    savings: List[float] = []
    for d in banked:
        b = d.bank()
        r = Racer(d.actions, cfg)
        for row in b:
            if r.resolved():
                break
            r.observe({a: row[a] for a in r.live})
        out = r.outcome(prefer=d.policy_action)
        savings.append(out.saving)
        if out.separated_at is None:
            never += 1
            hist["never"] = hist.get("never", 0) + 1
        else:
            at.append(out.separated_at)
            key = str(out.separated_at)
            hist[key] = hist.get(key, 0) + 1
    n = max(1, len(banked))
    return {
        "n": len(banked),
        "never_separated": never,
        "never_frac": round(never / n, 4),
        "median_rounds_to_separate": (int(np.median(at)) if at else None),
        "mean_rounds_to_separate": (round(float(np.mean(at)), 2) if at else None),
        "histogram": hist,
        "mean_within_race_saving": round(float(np.mean(savings)), 4) if savings else None,
    }


# ---------------------------------------------------------------------------
# the collector
# ---------------------------------------------------------------------------


def _discover(traces_dir: str, limit_battles: int) -> List[str]:
    out: List[str] = []
    for dirpath, _dirs, files in os.walk(traces_dir):
        for f in sorted(files):
            if f.endswith("_reconstruction.json"):
                stem = os.path.join(dirpath, f[: -len("_reconstruction.json")])
                if os.path.exists(stem + "_summary.json") and os.path.exists(stem + "_states.npz"):
                    out.append(stem)
    out.sort()
    return out[:limit_battles] if limit_battles else out


def collect(model, mappings, *, traces_dir: str, n_decisions: int, rounds: int, m_opp: int,
            search_impl: str, pool_packed, seed: int, per_battle: int,
            log=None) -> Tuple[List[BankedDecision], Price]:
    # FLUSHED, because this loop runs for tens of minutes and its output is normally redirected to
    # a file: block buffering makes a healthy run indistinguishable from a wedged one for the whole
    # first 8 KB, which on the first real invocation was mistaken for a hang in the pool loader.
    log = log or (lambda msg: print(msg, flush=True))
    from agents.training.obs_materializer import materialize_decisions
    from main.search_dividend.alpha import alpha_publication
    from utils.bridge.reconstruction import ReconstructionRecord, replay_battle

    cfg = SearchConfig(arm="honest", budget_s=1e9,
                       caps=WidthCaps(m_opp=m_opp, k_worlds=rounds, r_dice=1),
                       search_impl=search_impl, seed=seed)
    engine = SearchEngine(model=model, mappings=mappings, cfg=cfg, pool_packed=pool_packed)
    banked: List[BankedDecision] = []
    open_s: List[float] = []
    arm_s: List[float] = []
    rng = random.Random(seed)
    stems = _discover(traces_dir, 0)
    rng.shuffle(stems)
    log(f"[ab_racing] {len(stems)} candidate battles under {traces_dir}")

    for stem in stems:
        if len(banked) >= n_decisions:
            break
        try:
            record = ReconstructionRecord.load(stem + "_reconstruction.json")
            summary = json.load(open(stem + "_summary.json"))
            npz = np.load(stem + "_states.npz", allow_pickle=True)
            acts = np.asarray(npz["actions"], dtype=int)
            side = record.side_of(record.trainee_username)
            username = record.username(side)
            rep = replay_battle(record, impl=search_impl)
            our_full = rep.p1_chunks if side == "p1" else rep.p2_chunks
        except Exception as e:                       # noqa: BLE001
            log(f"  skip {os.path.basename(stem)}: {type(e).__name__}: {e}")
            continue

        invs = summary.get("invocations", [])
        cand = [i for i, iv in enumerate(invs)
                if iv.get("phase") == "move_selection" and i < len(acts)
                and int(np.asarray(npz["action_mask"][i]).sum()) >= 3]
        rng.shuffle(cand)
        taken = 0
        for i in sorted(cand[:per_battle]):
            if len(banked) >= n_decisions:
                break
            turn = int(invs[i]["turn"])
            try:
                trace = materialize_decisions(
                    our_full, username=username, packed_team=record.packed_team(side), side=side,
                    actions=[int(a) for a in acts[: i + 1]], battle_format=record.format_id,
                    battle_tag=record.battle_tag, mappings=mappings,
                    map_actions_at=i, stop_after_decision=i, encode_only_at=set())
                tokens = dict(trace.action_choices or {})
                if len(tokens) < 3 or int(acts[i]) not in tokens:
                    continue
                # The TRUE record's own root gives our side's protocol up to the decision — which
                # is exactly what the live player's builder accumulates, and exactly what the
                # world prefix gate is compared against.
                root = engine.session().open_root(turn, record=record)
                chunks = root.prefix_p1_chunks if side == "p1" else root.prefix_p2_chunks
                # 🚨 LINES, not chunks. The live player passes `LiveRecordBuilder.our_lines`, and
                # `determinize.revealed_species` reads one protocol line per element — handed a
                # chunk it sees only that chunk's FIRST line, reports NOTHING revealed, and the
                # determinizer then resamples all six opponent slots INCLUDING the lead. The
                # recorded turn-1 choice is then illegal for a mon that is not there, the replay
                # stalls before `|turn|1`, and every world dies as `open_root` "never reached the
                # start of turn N". Measured: 100% of worlds lost this way, against the live
                # battery's ~2.5% gate rate. `prefix_matches` tolerates either shape, which is
                # exactly why this was silent.
                observed = [ln for ch in chunks for ln in str(ch).split("\n")]
                pub = _alpha_at(model, npz, i, alpha_publication)
                t0 = time.monotonic()
                bank, meta = bank_decision(
                    engine, record=record, side=side, turn=turn,
                    our_history=[int(a) for a in acts[:i]], our_tokens=tokens,
                    observed_our_lines=observed, pub=pub, rounds=rounds, m_opp=m_opp)
                wall = time.monotonic() - t0
            except Exception as e:                   # noqa: BLE001
                log(f"  skip {os.path.basename(stem)}#{i}: {type(e).__name__}: {e}")
                continue
            if len(bank) < max(4, rounds // 2):
                log(f"  thin {os.path.basename(stem)}#{i}: {len(bank)}/{rounds} rounds "
                    f"(gate={meta['gate_failed']} drop={meta['dropped']} "
                    f"open_fail={meta['open_failed']} {meta.get('open_error', '')})")
                continue
            acts_sorted = sorted(tokens)
            banked.append(BankedDecision(
                battle=os.path.basename(stem), inv=i, turn=turn, actions=acts_sorted,
                policy_action=int(acts[i]), score_mode=meta["score_mode"],
                rounds_dropped=rounds - len(bank),
                rounds=[[row[a] for a in acts_sorted] for row in bank]))
            if meta["open_s"] > 0 and meta["slots"] > 0:
                # The two cost constants, MEASURED per decision and pooled by median. Both are
                # turn-dependent (the prefix replay grows with the turn number), so an assumed
                # constant would price the late game — where the decisions that matter live — at
                # the early game's rate.
                open_s.append(meta["open_s"] / max(1, meta["opens"]))
                arm_s.append(max(0.0, wall - meta["open_s"]) / meta["slots"])
            taken += 1
            log(f"  [{len(banked)}/{n_decisions}] {os.path.basename(stem)}#{i} t{turn} "
                f"A={len(acts_sorted)} rounds={len(bank)} {wall:.1f}s")
        del taken
    engine.close()
    price = Price(open_s=round(float(np.median(open_s)), 4) if open_s else Price().open_s,
                  arm_s=round(float(np.median(arm_s)), 5) if arm_s else Price().arm_s)
    return banked, price


def _alpha_at(model, npz, i: int, alpha_publication):
    """α off a forward of the RECORDED observation — the same publication the live player reads.

    Re-derived rather than stored because the trace keeps no α array; the obs is the same one the
    decision was made on, so the forward reproduces the stash exactly.
    """
    import torch

    with torch.no_grad():
        model.policy.predict_values({
            "observation": torch.as_tensor(np.asarray(npz["obs"][i])[None, :], dtype=torch.float32),
            "action_mask": torch.as_tensor(np.asarray(npz["action_mask"][i])[None, :],
                                           dtype=torch.float32)})
    return alpha_publication(getattr(model.policy, "features_extractor", None))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m main.search_dividend.ab_racing",
                                description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", nargs="?", help="run dir or checkpoint .zip (phase 1 only)")
    p.add_argument("--traces", help="eval_traces directory to sample decisions from")
    p.add_argument("--decisions", type=int, default=120)
    p.add_argument("--per-battle", type=int, default=3,
                   help="decisions sampled per battle — capped so the set is not one game's tail")
    p.add_argument("--rounds", type=int, default=32, help="paired samples banked per decision")
    p.add_argument("--max-opp", type=int, default=4, help="alpha-pruned opponent actions per round")
    p.add_argument("--bank", default="tmp/racing_bank.json")
    p.add_argument("--replay", metavar="BANK", help="phase 2 only: read a bank and report")
    p.add_argument("--out", help="write the phase-2 report JSON here")
    p.add_argument("--device", default="cpu")
    p.add_argument("--search-impl", default="rust", choices=["node", "rust"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pool-size", type=int, default=0)
    p.add_argument("--racing-rule", default=RacingConfig.rule, choices=list(RULES))
    p.add_argument("--racing-z", type=float, default=RacingConfig.z)
    p.add_argument("--racing-delta", type=float, default=RacingConfig.delta)
    p.add_argument("--racing-min-samples", type=int, default=RacingConfig.min_samples)
    p.add_argument("--budgets", default="",
                   help="comma-separated wall-clock seconds; default a geometric sweep")
    return p


def _default_budgets(price: Price, n_actions: float, max_rounds: int) -> List[float]:
    """A sweep from one full grid round up to the whole bank, geometric so the low end — where the
    allocators actually differ — is not swallowed by the high end."""
    lo = price.seconds(1, int(round(n_actions)))
    hi = price.seconds(max_rounds, int(round(n_actions)) * max_rounds)
    return [round(lo * (hi / lo) ** (k / 11.0), 4) for k in range(12)]


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = RacingConfig(rule=args.racing_rule, z=args.racing_z, delta=args.racing_delta,
                       min_samples=args.racing_min_samples)

    if args.replay:
        blob = json.load(open(args.replay))
        banked = [BankedDecision(**d) for d in blob["decisions"]]
        price = Price(**blob["price"])
        n_act = float(np.mean([len(d.actions) for d in banked])) if banked else 4.0
        max_r = max((len(d.rounds) for d in banked), default=8)
        budgets = ([float(x) for x in args.budgets.split(",") if x.strip()]
                   or _default_budgets(price, n_act, max_r))
        rep = replay(banked, price=price, cfg=cfg, budgets_s=budgets)
        rep["source"] = args.replay
        txt = json.dumps(rep, indent=2)
        if args.out:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            open(args.out, "w").write(txt)
        print(txt)
        return 0

    if not (args.model and args.traces):
        build_parser().print_usage(sys.stderr)
        print("error: phase 1 needs a model and --traces (or use --replay)", file=sys.stderr)
        return 2

    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    from agents.observation.state_encoder import load_mappings
    from main.search_dividend.__main__ import _load_model, _pool

    model, ckpt = _load_model(args.model, args.device)
    mappings = load_mappings()
    print(f"[ab_racing] ckpt={ckpt} rounds={args.rounds} m_opp={args.max_opp} "
          f"impl={args.search_impl}", flush=True)
    banked, price = collect(model, mappings, traces_dir=args.traces,
                            n_decisions=args.decisions, rounds=args.rounds, m_opp=args.max_opp,
                            search_impl=args.search_impl, pool_packed=_pool(args.pool_size),
                            seed=args.seed, per_battle=args.per_battle)
    os.makedirs(os.path.dirname(args.bank) or ".", exist_ok=True)
    with open(args.bank, "w") as fh:
        json.dump({"ckpt": ckpt, "price": asdict(price), "rounds": args.rounds,
                   "m_opp": args.max_opp,
                   "decisions": [d.as_dict() for d in banked]}, fh)
    print(f"[ab_racing] banked {len(banked)} decisions -> {args.bank}  price={price}", flush=True)
    n_act = float(np.mean([len(d.actions) for d in banked])) if banked else 4.0
    max_r = max((len(d.rounds) for d in banked), default=8)
    rep = replay(banked, price=price, cfg=cfg,
                 budgets_s=_default_budgets(price, n_act, max_r))
    txt = json.dumps(rep, indent=2)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        open(args.out, "w").write(txt)
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
