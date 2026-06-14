"""Human-agreement probe — how often does a trained policy match strong-ladder humans?

A *measure-before-you-build* diagnostic for the ai_v6 behavioural-cloning question
(memory: `feedback_research_state`). It replays saved spectator ladder ``.log`` files
(``replays/showdown/gen3ou/``) through the live obs pipeline (``agents.bc.log_reader``),
runs the trained policy on each reconstructed decision, and reports how often the
policy's argmax matches the human's action — plus *where* it diverges.

Why this answers the BC question
--------------------------------
The probe and BC share the **exact same** ``log → (obs, action, mask)`` pipeline, so
this both (a) validates that pipeline on real data before any training commits, and
(b) quantifies the upside: if the policy already agrees with strong humans, BC buys
little and the *divergences are the lever list*; if agreement is low, that measures how
far self-play has drifted off the human distribution.

Read the fidelity caveats in the output — a spectator log under-represents move
alternatives early-game (only revealed moves are legal) and cannot represent a switch
to an unrevealed own mon. The conditional breakdowns make those caveats quantitative
rather than hand-waved.

Usage
-----
    python -m main.human_agreement <run_dir> [--min-rating 1500] [--max-replays 400]
                                   [--replay-dir replays/showdown/gen3ou] [--device cpu]
                                   [--ckpt PATH] [--json OUT.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from agents.bc.log_reader import SpectatorLogReader
from agents.observation.state_encoder import load_mappings
from main.prober.model import ProbeModel

_PLAYER_RE = re.compile(r"^\|player\|(p[12])\|([^|]*)\|[^|]*\|(\d+)\s*$")


# --------------------------------------------------------------------------- #
# Checkpoint resolution                                                        #
# --------------------------------------------------------------------------- #
def resolve_checkpoint(run_dir: str, override: Optional[str]) -> str:
    if override:
        return override
    cand = os.path.join(run_dir, "best_model", "best_model.zip")
    if os.path.exists(cand):
        return cand
    cand = os.path.join(run_dir, "best_model.zip")
    if os.path.exists(cand):
        return cand
    # Current layout: <run>/checkpoints/; legacy: <run>/ root. Search both.
    ckpts = glob.glob(os.path.join(run_dir, "checkpoint_*_steps.zip")) + glob.glob(
        os.path.join(run_dir, "checkpoints", "checkpoint_*_steps.zip")
    )
    if ckpts:
        def _steps(p: str) -> int:
            m = re.search(r"checkpoint_(\d+)_steps", p)
            return int(m.group(1)) if m else 0
        return max(ckpts, key=_steps)
    if run_dir.endswith(".zip") and os.path.exists(run_dir):
        return run_dir
    raise FileNotFoundError(f"no checkpoint found under {run_dir!r}")


def strong_sides(log_text: str, min_rating: int) -> List[Tuple[str, str, int]]:
    """Return [(role, username, rating)] for sides whose ladder rating >= min_rating."""
    out: List[Tuple[str, str, int]] = []
    for line in log_text.split("\n"):
        m = _PLAYER_RE.match(line)
        if m:
            role, name, rating = m.group(1), m.group(2), int(m.group(3))
            if rating >= min_rating and name and not any(o[0] == role for o in out):
                out.append((role, name, rating))
        if line.startswith("|turn|2"):
            break
    return out


# --------------------------------------------------------------------------- #
# Aggregation                                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class Bucket:
    n: int = 0
    agree: int = 0
    p_human_sum: float = 0.0

    def add(self, hit: bool, p_human: float) -> None:
        self.n += 1
        self.agree += int(hit)
        self.p_human_sum += p_human

    def as_dict(self) -> dict:
        if self.n == 0:
            return {"n": 0, "agreement": None, "mean_p_human": None}
        return {
            "n": self.n,
            "agreement": round(self.agree / self.n, 4),
            "mean_p_human": round(self.p_human_sum / self.n, 4),
        }


@dataclass
class Aggregator:
    overall: Bucket = field(default_factory=Bucket)
    move: Bucket = field(default_factory=Bucket)
    switch: Bucket = field(default_factory=Bucket)
    # move agreement conditioned on how many of the active's moves are revealed (legal)
    by_n_move: Dict[int, Bucket] = field(default_factory=lambda: defaultdict(Bucket))
    by_turn: Dict[str, Bucket] = field(default_factory=lambda: defaultdict(Bucket))
    by_rating: Dict[str, Bucket] = field(default_factory=lambda: defaultdict(Bucket))
    # under-switching cross-tab: rows = what the human did, cols = what the policy did
    human_switch_policy_switch: int = 0
    human_switch_policy_move: int = 0
    human_move_policy_switch: int = 0
    human_move_policy_move: int = 0
    policy_switch_mass_sum: float = 0.0   # Σ policy P(any switch), for the decisions where switching was legal
    switchable_decisions: int = 0
    # fidelity
    replays: int = 0
    sides: int = 0
    excl_switch_unrevealed: int = 0
    excl_other: int = 0

    @staticmethod
    def _turn_bucket(t: int) -> str:
        if t <= 5:
            return "01-05"
        if t <= 10:
            return "06-10"
        if t <= 20:
            return "11-20"
        return "21+"

    @staticmethod
    def _rating_bucket(r: int) -> str:
        if r >= 1700:
            return "1700+"
        if r >= 1600:
            return "1600-1699"
        if r >= 1500:
            return "1500-1599"
        return "<1500"

    def add_decision(self, d, pred: int, probs: np.ndarray, rating: int) -> None:
        hit = (pred == d.human_action)
        p_human = float(probs[d.human_action])
        self.overall.add(hit, p_human)
        self._turn_bucket  # noqa
        self.by_turn[self._turn_bucket(d.turn)].add(hit, p_human)
        self.by_rating[self._rating_bucket(rating)].add(hit, p_human)

        n_move_legal = int(d.mask[6:10].sum())
        n_switch_legal = int(d.mask[0:6].sum())
        policy_is_switch = pred < 6
        human_is_switch = d.action_type == "switch"

        if d.action_type == "move":
            self.move.add(hit, p_human)
            self.by_n_move[n_move_legal].add(hit, p_human)
        else:
            self.switch.add(hit, p_human)

        if human_is_switch and policy_is_switch:
            self.human_switch_policy_switch += 1
        elif human_is_switch and not policy_is_switch:
            self.human_switch_policy_move += 1
        elif (not human_is_switch) and policy_is_switch:
            self.human_move_policy_switch += 1
        else:
            self.human_move_policy_move += 1

        if n_switch_legal > 0:
            self.switchable_decisions += 1
            self.policy_switch_mass_sum += float(probs[0:6].sum())

    def report(self) -> dict:
        hs = self.human_switch_policy_switch + self.human_switch_policy_move
        hm = self.human_move_policy_switch + self.human_move_policy_move
        human_switch_rate = (hs / (hs + hm)) if (hs + hm) else None
        policy_switch_rate = (
            (self.human_switch_policy_switch + self.human_move_policy_switch) / (hs + hm)
            if (hs + hm) else None
        )
        return {
            "headline": {
                "overall": self.overall.as_dict(),
                "move": self.move.as_dict(),
                "switch": self.switch.as_dict(),
            },
            "under_switching": {
                "human_switch_rate": round(human_switch_rate, 4) if human_switch_rate is not None else None,
                "policy_switch_rate": round(policy_switch_rate, 4) if policy_switch_rate is not None else None,
                "mean_policy_switch_prob_when_switch_legal": (
                    round(self.policy_switch_mass_sum / self.switchable_decisions, 4)
                    if self.switchable_decisions else None
                ),
                "crosstab": {
                    "human_switch__policy_switch": self.human_switch_policy_switch,
                    "human_switch__policy_move": self.human_switch_policy_move,
                    "human_move__policy_switch": self.human_move_policy_switch,
                    "human_move__policy_move": self.human_move_policy_move,
                },
                "note": "human_switch_rate vs policy_switch_rate over the SAME states is "
                        "the under-switching test; switch mask = revealed-alive bench only.",
            },
            "move_agreement_by_revealed_moves": {
                str(k): self.by_n_move[k].as_dict() for k in sorted(self.by_n_move)
            },
            "by_turn": {k: self.by_turn[k].as_dict() for k in sorted(self.by_turn)},
            "by_rating": {k: self.by_rating[k].as_dict() for k in sorted(self.by_rating)},
            "fidelity": {
                "replays_used": self.replays,
                "sides_reconstructed": self.sides,
                "decisions_scored": self.overall.n,
                "excluded_switch_to_unrevealed_own_mon": self.excl_switch_unrevealed,
                "excluded_other": self.excl_other,
                "caveats": [
                    "Spectator logs carry no |request| JSON: move alternatives are "
                    "under-represented early-game (only revealed moves are legal), which "
                    "inflates move agreement at low revealed-move counts — see "
                    "move_agreement_by_revealed_moves (4 = most faithful).",
                    "Own EV/IV spreads are absent and own bench is known only as revealed; "
                    "switches to a never-revealed own mon are excluded, so switch behaviour "
                    "is only partially reconstructable (skews to mid/late game).",
                    "These limits are intrinsic to spectator logs and apply to BC too — the "
                    "probe measures the real conditions BC would train under.",
                ],
            },
        }


# --------------------------------------------------------------------------- #
# Driver                                                                        #
# --------------------------------------------------------------------------- #
def run(run_dir: str, *, min_rating: int, max_replays: int, replay_dir: str,
        device: str, ckpt_override: Optional[str], threads: int = 4) -> dict:
    # Be a good neighbour: a live training run may own most cores, so cap torch
    # intra-op threads rather than contending for all of them (batched forwards
    # recover the throughput).
    try:
        import torch
        torch.set_num_threads(max(1, threads))
    except Exception:
        pass
    ckpt = resolve_checkpoint(run_dir, ckpt_override)
    mappings = load_mappings()
    reader = SpectatorLogReader(mappings)
    model = ProbeModel.load(ckpt, device=device)

    files = sorted(glob.glob(os.path.join(replay_dir, "**", "*.log"), recursive=True))
    agg = Aggregator()
    used = 0
    for fp in files:
        if used >= max_replays:
            break
        try:
            text = open(fp, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        sides = strong_sides(text, min_rating)
        if not sides:
            continue
        contributed = False
        for role, name, rating in sides:
            try:
                res = reader.read(text, our_username=name)
            except Exception:  # malformed log / parse edge — skip the side, keep going
                continue
            agg.excl_switch_unrevealed += res.stats.excluded_switch_unrevealed
            agg.excl_other += res.stats.excluded_move_unmapped + res.stats.excluded_empty_mask
            if not res.decisions:
                continue
            obs = np.stack([d.obs for d in res.decisions])
            masks = np.stack([d.mask for d in res.decisions])
            probs_batch = model.action_probs_batch(obs, masks)   # one forward per replay
            for i, d in enumerate(res.decisions):
                probs = probs_batch[i]
                agg.add_decision(d, int(np.argmax(probs)), probs, rating)
            agg.sides += 1
            contributed = True
        if contributed:
            used += 1
    agg.replays = used
    return agg.report()


def _print_summary(rep: dict) -> None:
    h = rep["headline"]
    f = rep["fidelity"]
    us = rep["under_switching"]
    print("\n================  HUMAN-AGREEMENT PROBE  ================")
    print(f"replays={f['replays_used']}  sides={f['sides_reconstructed']}  "
          f"decisions={f['decisions_scored']}")
    print(f"\n  OVERALL agreement : {_fmt(h['overall'])}")
    print(f"  MOVE    agreement : {_fmt(h['move'])}")
    print(f"  SWITCH  agreement : {_fmt(h['switch'])}")
    print(f"\n  under-switching:  human_switch_rate={us['human_switch_rate']}  "
          f"policy_switch_rate={us['policy_switch_rate']}  "
          f"mean_policy_switch_prob={us['mean_policy_switch_prob_when_switch_legal']}")
    print("\n  move agreement by # revealed (legal) moves  [4 = most faithful]:")
    for k, v in rep["move_agreement_by_revealed_moves"].items():
        print(f"    {k} legal moves : {_fmt(v)}")
    print("\n  by turn:")
    for k, v in rep["by_turn"].items():
        print(f"    turn {k:6s}: {_fmt(v)}")
    print("\n  by rating bucket:")
    for k, v in rep["by_rating"].items():
        print(f"    {k:10s}: {_fmt(v)}")
    print(f"\n  fidelity: excluded switch-to-unrevealed={f['excluded_switch_to_unrevealed_own_mon']}  "
          f"other={f['excluded_other']}")
    print("========================================================\n")


def _fmt(b: dict) -> str:
    if not b or b.get("n", 0) == 0:
        return "n=0"
    return f"n={b['n']:<5d} agreement={b['agreement']:.1%}  mean_p_human={b['mean_p_human']:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="model run dir (best_model/ resolved) or a .zip")
    ap.add_argument("--min-rating", type=int, default=1500,
                    help="only reconstruct sides with ladder rating >= this (default 1500)")
    ap.add_argument("--max-replays", type=int, default=400,
                    help="cap on replays that contribute a side (default 400)")
    ap.add_argument("--replay-dir", default="replays/showdown/gen3ou",
                    help="root of the .log corpus")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4,
                    help="torch intra-op threads (keep low to spare a live training run)")
    ap.add_argument("--ckpt", default=None, help="explicit checkpoint .zip (overrides run_dir)")
    ap.add_argument("--json", default=None, help="write the full JSON report to this path")
    args = ap.parse_args()

    rep = run(args.run_dir, min_rating=args.min_rating, max_replays=args.max_replays,
              replay_dir=args.replay_dir, device=args.device, ckpt_override=args.ckpt,
              threads=args.threads)
    _print_summary(rep)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rep, fh, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
